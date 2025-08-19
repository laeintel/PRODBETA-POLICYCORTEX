'use client';

import React, { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import NextLink from 'next/link';
import { 
  Activity, BarChart, Shield, Briefcase, Server, Cpu, DollarSign,
  GitBranch, MessageSquare, Settings, ChevronDown, ChevronLeft,
  ChevronRight, Power, Grid, Heart, AlertTriangle, Upload, Archive,
  Zap, Terminal, Gauge, CheckCircle, TrendingUp, LineChart, FileText,
  Bell, XOctagon, ScanLine, Lock, UserCheck, FileCheck, Award, Users,
  Layout, Percent, Tag, FileCode, BookOpen, FileX, Target, Building2,
  HardDrive, Network, Layers, Globe, AtSign, Package, Hexagon, Monitor,
  Database, Inbox, AlertCircle, Link2, Wallet, Share2, CreditCard,
  Calendar, Star, Folder, Smartphone, Radio, Download, Save
} from 'lucide-react';

interface SidebarItem {
  label: string;
  path: string;
  icon: any;
  badge?: string;
}

interface SidebarSection {
  id: string;
  title: string;
  icon: any;
  items: SidebarItem[];
}

export default function TacticalSidebar() {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  // Load sidebar state from localStorage
  useEffect(() => {
    const savedState = localStorage.getItem('sidebar-state');
    if (savedState) {
      const { open, expanded } = JSON.parse(savedState);
      setSidebarOpen(open);
      setExpandedSection(expanded);
    } else {
      // Auto-expand section based on current path
      const section = detectSectionFromPath(pathname);
      setExpandedSection(section);
    }
  }, []);

  // Save sidebar state to localStorage
  useEffect(() => {
    localStorage.setItem('sidebar-state', JSON.stringify({
      open: sidebarOpen,
      expanded: expandedSection
    }));
  }, [sidebarOpen, expandedSection]);

  // Auto-expand section when navigating
  useEffect(() => {
    const section = detectSectionFromPath(pathname);
    if (section && section !== expandedSection) {
      setExpandedSection(section);
    }
  }, [pathname]);

  const detectSectionFromPath = (path: string): string | null => {
    // Check for Operations items first - these should stay in Operations section
    if (path === '/tactical' || path === '/tactical/monitor' || path === '/tactical/resources' || 
        path === '/tactical/health' || path === '/tactical/incidents' || 
        path === '/tactical/changes' || path === '/tactical/deploy' || 
        path === '/tactical/backup' || path === '/tactical/automation' || 
        path === '/tactical/workflow') {
      return 'operations';
    }
    // Monitoring section paths
    if (path.includes('/monitoring-overview') || path.includes('/metrics') || path.includes('/logs') ||
        path.includes('/alerts') || path.includes('/events') ||
        path.includes('/availability') || path.includes('/capacity') || path.includes('/predictive') ||
        path.includes('/reports') || path.includes('/traces')) {
      return 'monitoring';
    }
    if (path.includes('/security') || path.includes('/threats') || path.includes('/compliance') || 
        path.includes('/policies') || path.includes('/audit') || path.includes('/vulnerabilities')) {
      return 'security';
    }
    if (path.includes('/governance') || path.includes('/policy-dashboard') || path.includes('/risk') || 
        path.includes('/cost-governance') || path.includes('/tags') || path.includes('/exemptions')) {
      return 'governance';
    }
    if (path.includes('/infrastructure') || path.includes('/compute') || path.includes('/storage') || 
        path.includes('/network') || path.includes('/kubernetes') || path.includes('/vms')) {
      return 'infrastructure';
    }
    if (path.includes('/ai') || path.includes('/training') || path.includes('/predictions') || 
        path.includes('/anomalies') || path.includes('/correlation') || path.includes('/models')) {
      return 'intelligence';
    }
    if (path.includes('/cost') || path.includes('/budgets') || path.includes('/invoices') || 
        path.includes('/optimization') || path.includes('/forecast') || path.includes('/savings')) {
      return 'financial';
    }
    if (path.includes('/pipelines') || path.includes('/builds') || path.includes('/deployments') || 
        path.includes('/releases') || path.includes('/repos')) {
      return 'devops';
    }
    if (path.includes('/notifications') || path.includes('/emails') || path.includes('/webhooks') || 
        path.includes('/slack') || path.includes('/broadcast')) {
      return 'communication';
    }
    if (path.includes('/settings') || path.includes('/users') || path.includes('/roles') || 
        path.includes('/api-keys') || path.includes('/integrations')) {
      return 'administration';
    }
    return 'operations'; // default
  };

  const toggleSection = (sectionId: string) => {
    // Only one section can be expanded at a time
    setExpandedSection(expandedSection === sectionId ? null : sectionId);
  };

  const sidebarSections: SidebarSection[] = [
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
        { label: 'Monitoring Overview', path: '/tactical/monitoring-overview', icon: Activity },
        { label: 'Metrics Dashboard', path: '/tactical/metrics', icon: BarChart },
        { label: 'Log Analytics', path: '/tactical/logs', icon: Terminal },
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
        { label: 'Security Overview', path: '/tactical/security', icon: Shield, badge: '3' },
        { label: 'Threat Detection', path: '/tactical/threats', icon: XOctagon },
        { label: 'Vulnerability Scan', path: '/tactical/vulnerabilities', icon: ScanLine },
        { label: 'Access Control', path: '/tactical/access', icon: Lock },
        { label: 'Identity Management', path: '/tactical/identity', icon: UserCheck },
        { label: 'Compliance Hub', path: '/tactical/compliance', icon: CheckCircle },
        { label: 'Policy Engine', path: '/tactical/policies', icon: Briefcase },
        { label: 'Audit Trail', path: '/tactical/audit', icon: FileCheck },
        { label: 'Encryption Keys', path: '/tactical/encryption', icon: Lock },
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
        { label: 'Governance Overview', path: '/tactical/policy-dashboard', icon: Layout },
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
        { label: 'Infrastructure Overview', path: '/tactical/infrastructure', icon: Activity },
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
        { label: 'Finance Overview', path: '/tactical/cost-governance', icon: DollarSign },
        { label: 'Cost Analytics', path: '/tactical/cost', icon: DollarSign },
        { label: 'Budget Tracking', path: '/tactical/budgets', icon: BarChart },
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
        { label: 'Test Results', path: '/tactical/tests', icon: CheckCircle },
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
        { label: 'Email Templates', path: '/tactical/emails', icon: FileText },
        { label: 'SMS Alerts', path: '/tactical/sms', icon: Smartphone },
        { label: 'Webhooks', path: '/tactical/webhooks', icon: Link2 },
        { label: 'Slack Integration', path: '/tactical/slack', icon: MessageSquare },
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
        { label: 'API Keys', path: '/tactical/api-keys', icon: Lock },
        { label: 'Integrations', path: '/tactical/integrations', icon: Link2 },
        { label: 'Backup Config', path: '/tactical/backup-config', icon: Save },
        { label: 'System Logs', path: '/tactical/system-logs', icon: Terminal },
        { label: 'License Manager', path: '/tactical/licenses', icon: CreditCard },
        { label: 'Update Center', path: '/tactical/updates', icon: Download },
        { label: 'Documentation', path: '/tactical/docs', icon: BookOpen }
      ]
    }
  ];

  return (
    <div className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-gray-900 border-r border-gray-800 transition-all duration-300 flex flex-col h-screen sticky top-0`}>
      {/* Sidebar Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className={`flex items-center space-x-3 ${!sidebarOpen && 'justify-center'}`}>
            <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-green-500 rounded flex items-center justify-center font-bold text-white">
              PC
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="text-sm font-bold">POLICYCORTEX</h1>
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
        {sidebarSections.map((section) => {
          const isExpanded = expandedSection === section.id;
          const SectionIcon = section.icon;
          
          return (
            <div key={section.id} className="border-b border-gray-800">
              <button
                onClick={() => toggleSection(section.id)}
                className={`w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800 transition-colors ${
                  isExpanded ? 'bg-gray-800/50' : ''
                }`}
              >
                <div className="flex items-center space-x-3">
                  <SectionIcon className="w-4 h-4 text-gray-400" />
                  {sidebarOpen && (
                    <span className="text-xs font-bold text-gray-400">{section.title}</span>
                  )}
                </div>
                {sidebarOpen && (
                  <ChevronDown className={`w-3 h-3 text-gray-500 transition-transform ${
                    isExpanded ? 'rotate-180' : ''
                  }`} />
                )}
              </button>
              
              {isExpanded && sidebarOpen && (
                <div className="pb-2">
                  {section.items.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.path;
                    
                    return (
                      <NextLink
                        key={item.path}
                        href={item.path}
                        className={`flex items-center justify-between px-4 py-2 text-xs hover:bg-gray-800 transition-colors group ${
                          isActive ? 'bg-gray-800 border-l-2 border-green-500' : ''
                        }`}
                      >
                        <div className="flex items-center space-x-3">
                          <Icon className={`w-3 h-3 ${
                            isActive ? 'text-green-500' : 'text-gray-500 group-hover:text-gray-300'
                          }`} />
                          <span className={`${
                            isActive ? 'text-white' : 'text-gray-400 group-hover:text-gray-200'
                          }`}>
                            {item.label}
                          </span>
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
          );
        })}
      </div>

      {/* Sidebar Footer */}
      <div className="p-4 border-t border-gray-800">
        <button className="w-full px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded transition-colors flex items-center justify-center space-x-2">
          <Power className="w-3 h-3" />
          {sidebarOpen && <span>EMERGENCY SHUTDOWN</span>}
        </button>
      </div>
    </div>
  );
}