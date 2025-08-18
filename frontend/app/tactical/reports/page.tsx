'use client';

import React, { useState, useEffect } from 'react';
import { 
  FileText, Plus, Search, Filter, Calendar, Download, Settings,
  Play, Pause, Clock, BarChart, PieChart, LineChart, Table,
  Edit, Trash2, Copy, Share, Eye, ChevronRight, Star, StarOff,
  RefreshCw, Archive, Bookmark, Send, Users, Globe, Database,
  TrendingUp, AlertCircle, CheckCircle, XCircle, Info
} from 'lucide-react';
import { api } from '../../../lib/api-client';

interface CustomReport {
  id: string;
  name: string;
  description: string;
  category: 'performance' | 'availability' | 'security' | 'capacity' | 'compliance' | 'business';
  type: 'dashboard' | 'table' | 'chart' | 'summary' | 'detailed';
  status: 'active' | 'draft' | 'archived' | 'scheduled';
  favorite: boolean;
  author: string;
  lastRun: string;
  nextRun?: string;
  frequency: 'manual' | 'hourly' | 'daily' | 'weekly' | 'monthly';
  recipients: string[];
  dataSource: string[];
  filters: {
    timeRange: string;
    services?: string[];
    environment?: string;
    [key: string]: any;
  };
  visualizations: {
    type: 'line' | 'bar' | 'pie' | 'table' | 'metric' | 'gauge';
    title: string;
    dataQuery: string;
    position: { x: number; y: number; w: number; h: number; };
  }[];
  permissions: {
    view: string[];
    edit: string[];
    delete: string[];
  };
  metadata: {
    created: string;
    modified: string;
    version: string;
    tags: string[];
    executions: number;
  };
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  thumbnail: string;
  complexity: 'simple' | 'intermediate' | 'advanced';
  estimatedTime: string;
  includes: string[];
  previewData: any;
}

interface ReportExecution {
  id: string;
  reportId: string;
  reportName: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  startTime: string;
  endTime?: string;
  duration?: number;
  output?: {
    format: 'pdf' | 'excel' | 'csv' | 'json';
    size: number;
    downloadUrl: string;
  };
  error?: string;
  triggeredBy: string;
}

export default function CustomReports() {
  const [reports, setReports] = useState<CustomReport[]>([]);
  const [templates, setTemplates] = useState<ReportTemplate[]>([]);
  const [executions, setExecutions] = useState<ReportExecution[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'reports' | 'templates' | 'executions' | 'builder'>('reports');
  const [showFavorites, setShowFavorites] = useState(false);
  const [selectedReport, setSelectedReport] = useState<CustomReport | null>(null);

  useEffect(() => {
    // Initialize with mock reports data
    setReports([
      {
        id: 'RPT-001',
        name: 'Weekly Performance Summary',
        description: 'Comprehensive performance metrics across all services',
        category: 'performance',
        type: 'dashboard',
        status: 'active',
        favorite: true,
        author: 'admin@company.com',
        lastRun: '2 hours ago',
        nextRun: 'In 5 days',
        frequency: 'weekly',
        recipients: ['team-leads@company.com', 'management@company.com'],
        dataSource: ['metrics-db', 'monitoring-api', 'logs-service'],
        filters: {
          timeRange: '7d',
          services: ['api-gateway', 'auth-service', 'database'],
          environment: 'production'
        },
        visualizations: [
          {
            type: 'line',
            title: 'Response Time Trends',
            dataQuery: 'SELECT avg(response_time) FROM metrics WHERE service IN (?)',
            position: { x: 0, y: 0, w: 6, h: 3 }
          },
          {
            type: 'gauge',
            title: 'Average Availability',
            dataQuery: 'SELECT avg(uptime) FROM availability',
            position: { x: 6, y: 0, w: 3, h: 3 }
          },
          {
            type: 'bar',
            title: 'Error Rates by Service',
            dataQuery: 'SELECT service, error_rate FROM service_metrics',
            position: { x: 9, y: 0, w: 3, h: 3 }
          }
        ],
        permissions: {
          view: ['all-users'],
          edit: ['admin', 'report-managers'],
          delete: ['admin']
        },
        metadata: {
          created: '2024-01-15',
          modified: '2024-01-20',
          version: '1.3',
          tags: ['performance', 'weekly', 'management'],
          executions: 24
        }
      },
      {
        id: 'RPT-002',
        name: 'Security Incident Analysis',
        description: 'Monthly security incidents and threat analysis',
        category: 'security',
        type: 'detailed',
        status: 'active',
        favorite: false,
        author: 'security@company.com',
        lastRun: '1 day ago',
        nextRun: 'In 29 days',
        frequency: 'monthly',
        recipients: ['security-team@company.com', 'ciso@company.com'],
        dataSource: ['security-logs', 'incident-db', 'threat-intel'],
        filters: {
          timeRange: '30d',
          severity: ['high', 'critical'],
          resolved: true
        },
        visualizations: [
          {
            type: 'table',
            title: 'Incident Summary',
            dataQuery: 'SELECT * FROM security_incidents',
            position: { x: 0, y: 0, w: 12, h: 4 }
          },
          {
            type: 'pie',
            title: 'Threat Categories',
            dataQuery: 'SELECT category, COUNT(*) FROM threats',
            position: { x: 0, y: 4, w: 6, h: 3 }
          }
        ],
        permissions: {
          view: ['security-team'],
          edit: ['security-admin'],
          delete: ['security-admin']
        },
        metadata: {
          created: '2024-01-10',
          modified: '2024-01-18',
          version: '2.1',
          tags: ['security', 'incidents', 'monthly'],
          executions: 6
        }
      },
      {
        id: 'RPT-003',
        name: 'Capacity Planning Forecast',
        description: 'Resource utilization trends and capacity predictions',
        category: 'capacity',
        type: 'chart',
        status: 'active',
        favorite: true,
        author: 'ops@company.com',
        lastRun: '6 hours ago',
        frequency: 'daily',
        recipients: ['ops-team@company.com'],
        dataSource: ['capacity-metrics', 'resource-usage'],
        filters: {
          timeRange: '14d',
          resources: ['cpu', 'memory', 'storage'],
          threshold: 80
        },
        visualizations: [
          {
            type: 'line',
            title: 'CPU Utilization Trend',
            dataQuery: 'SELECT timestamp, avg(cpu_usage) FROM metrics',
            position: { x: 0, y: 0, w: 6, h: 3 }
          },
          {
            type: 'bar',
            title: 'Storage Growth',
            dataQuery: 'SELECT date, storage_used FROM capacity',
            position: { x: 6, y: 0, w: 6, h: 3 }
          }
        ],
        permissions: {
          view: ['ops-team', 'management'],
          edit: ['ops-admin'],
          delete: ['ops-admin']
        },
        metadata: {
          created: '2024-01-12',
          modified: '2024-01-19',
          version: '1.5',
          tags: ['capacity', 'forecasting', 'daily'],
          executions: 45
        }
      },
      {
        id: 'RPT-004',
        name: 'Compliance Audit Report',
        description: 'Quarterly compliance status and audit findings',
        category: 'compliance',
        type: 'summary',
        status: 'draft',
        favorite: false,
        author: 'compliance@company.com',
        lastRun: 'Never',
        frequency: 'manual',
        recipients: ['audit-team@company.com', 'legal@company.com'],
        dataSource: ['audit-logs', 'compliance-db', 'policy-engine'],
        filters: {
          timeRange: '90d',
          standards: ['SOX', 'GDPR', 'ISO27001'],
          status: ['all']
        },
        visualizations: [
          {
            type: 'metric',
            title: 'Compliance Score',
            dataQuery: 'SELECT compliance_percentage FROM audit_summary',
            position: { x: 0, y: 0, w: 3, h: 2 }
          },
          {
            type: 'table',
            title: 'Findings Summary',
            dataQuery: 'SELECT * FROM compliance_findings',
            position: { x: 3, y: 0, w: 9, h: 4 }
          }
        ],
        permissions: {
          view: ['compliance-team'],
          edit: ['compliance-admin'],
          delete: ['compliance-admin']
        },
        metadata: {
          created: '2024-01-22',
          modified: '2024-01-22',
          version: '0.1',
          tags: ['compliance', 'audit', 'quarterly'],
          executions: 0
        }
      }
    ]);

    setTemplates([
      {
        id: 'TPL-001',
        name: 'Service Health Dashboard',
        description: 'Monitor service availability and performance metrics',
        category: 'Performance',
        thumbnail: '/templates/service-health.png',
        complexity: 'simple',
        estimatedTime: '5 minutes',
        includes: ['Uptime metrics', 'Response times', 'Error rates', 'SLA compliance'],
        previewData: {}
      },
      {
        id: 'TPL-002',
        name: 'Executive Summary',
        description: 'High-level KPIs and business metrics for leadership',
        category: 'Business',
        thumbnail: '/templates/executive.png',
        complexity: 'intermediate',
        estimatedTime: '10 minutes',
        includes: ['Key metrics', 'Trend analysis', 'Cost optimization', 'Risk indicators'],
        previewData: {}
      },
      {
        id: 'TPL-003',
        name: 'Security Posture Report',
        description: 'Comprehensive security metrics and threat analysis',
        category: 'Security',
        thumbnail: '/templates/security.png',
        complexity: 'advanced',
        estimatedTime: '15 minutes',
        includes: ['Threat intelligence', 'Vulnerability assessment', 'Incident response', 'Compliance status'],
        previewData: {}
      }
    ]);

    setExecutions([
      {
        id: 'EXE-001',
        reportId: 'RPT-001',
        reportName: 'Weekly Performance Summary',
        status: 'completed',
        startTime: '2024-01-22T10:30:00Z',
        endTime: '2024-01-22T10:32:15Z',
        duration: 135,
        output: {
          format: 'pdf',
          size: 2.4,
          downloadUrl: '/reports/weekly-performance-20240122.pdf'
        },
        triggeredBy: 'scheduler'
      },
      {
        id: 'EXE-002',
        reportId: 'RPT-003',
        reportName: 'Capacity Planning Forecast',
        status: 'running',
        startTime: '2024-01-22T14:15:00Z',
        triggeredBy: 'admin@company.com'
      },
      {
        id: 'EXE-003',
        reportId: 'RPT-002',
        reportName: 'Security Incident Analysis',
        status: 'failed',
        startTime: '2024-01-21T08:00:00Z',
        endTime: '2024-01-21T08:05:30Z',
        duration: 330,
        error: 'Connection timeout to security-logs database',
        triggeredBy: 'scheduler'
      }
    ]);
  }, []);

  const getCategoryIcon = (category: string) => {
    switch(category) {
      case 'performance': return <BarChart className="w-4 h-4 text-blue-500" />;
      case 'availability': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'security': return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'capacity': return <Database className="w-4 h-4 text-purple-500" />;
      case 'compliance': return <FileText className="w-4 h-4 text-orange-500" />;
      case 'business': return <TrendingUp className="w-4 h-4 text-cyan-500" />;
      default: return <FileText className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-500 bg-green-900/20';
      case 'draft': return 'text-yellow-500 bg-yellow-900/20';
      case 'archived': return 'text-gray-500 bg-gray-900/20';
      case 'scheduled': return 'text-blue-500 bg-blue-900/20';
      case 'running': return 'text-blue-500 bg-blue-900/20';
      case 'completed': return 'text-green-500 bg-green-900/20';
      case 'failed': return 'text-red-500 bg-red-900/20';
      case 'queued': return 'text-yellow-500 bg-yellow-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch(type) {
      case 'dashboard': return <BarChart className="w-4 h-4" />;
      case 'table': return <Table className="w-4 h-4" />;
      case 'chart': return <LineChart className="w-4 h-4" />;
      case 'summary': return <FileText className="w-4 h-4" />;
      case 'detailed': return <Eye className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  const filteredReports = reports.filter(report => {
    if (showFavorites && !report.favorite) return false;
    if (selectedCategory !== 'all' && report.category !== selectedCategory) return false;
    if (selectedStatus !== 'all' && report.status !== selectedStatus) return false;
    if (searchQuery && !report.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const stats = {
    totalReports: reports.length,
    activeReports: reports.filter(r => r.status === 'active').length,
    scheduledReports: reports.filter(r => r.frequency !== 'manual').length,
    favoriteReports: reports.filter(r => r.favorite).length,
    runningExecutions: executions.filter(e => e.status === 'running').length,
    completedToday: executions.filter(e => e.status === 'completed' && new Date(e.startTime).toDateString() === new Date().toDateString()).length
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Custom Reports</h1>
            <p className="text-sm text-gray-400 mt-1">Create, manage, and distribute custom reports</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowFavorites(!showFavorites)}
              className={`px-3 py-2 rounded text-sm flex items-center space-x-2 ${
                showFavorites ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-gray-800 hover:bg-gray-700'
              }`}
            >
              {showFavorites ? <Star className="w-4 h-4" /> : <StarOff className="w-4 h-4" />}
              <span>Favorites</span>
            </button>
            
            <button
              onClick={() => setViewMode(
                viewMode === 'reports' ? 'templates' : 
                viewMode === 'templates' ? 'executions' : 
                viewMode === 'executions' ? 'builder' : 'reports'
              )}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm"
            >
              {viewMode === 'reports' ? 'Templates' : 
               viewMode === 'templates' ? 'Executions' : 
               viewMode === 'executions' ? 'Builder' : 'Reports'}
            </button>
            
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm flex items-center space-x-2">
              <Plus className="w-4 h-4" />
              <span>New Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="flex items-center space-x-3">
            <FileText className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Total Reports</p>
              <p className="text-xl font-bold">{stats.totalReports}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Active</p>
              <p className="text-xl font-bold">{stats.activeReports}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Clock className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Scheduled</p>
              <p className="text-xl font-bold">{stats.scheduledReports}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Star className="w-5 h-5 text-yellow-500" />
            <div>
              <p className="text-xs text-gray-400">Favorites</p>
              <p className="text-xl font-bold">{stats.favoriteReports}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Play className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-xs text-gray-400">Running</p>
              <p className="text-xl font-bold text-blue-500">{stats.runningExecutions}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Download className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-xs text-gray-400">Completed Today</p>
              <p className="text-xl font-bold">{stats.completedToday}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'reports' && (
          <>
            {/* Search and Filters */}
            <div className="flex items-center space-x-3 mb-6">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search reports..."
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
                />
              </div>
              
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Categories</option>
                <option value="performance">Performance</option>
                <option value="availability">Availability</option>
                <option value="security">Security</option>
                <option value="capacity">Capacity</option>
                <option value="compliance">Compliance</option>
                <option value="business">Business</option>
              </select>
              
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="draft">Draft</option>
                <option value="archived">Archived</option>
                <option value="scheduled">Scheduled</option>
              </select>
            </div>

            {/* Reports Grid */}
            <div className="grid grid-cols-2 gap-4">
              {filteredReports.map(report => (
                <div key={report.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      <div className="p-2 bg-gray-800 rounded">
                        {getCategoryIcon(report.category)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-sm font-bold">{report.name}</h3>
                          {report.favorite && <Star className="w-3 h-3 text-yellow-500" />}
                        </div>
                        <p className="text-xs text-gray-400 mb-2">{report.description}</p>
                        <div className="flex items-center space-x-2 text-xs text-gray-500">
                          <span>{report.category}</span>
                          <span>•</span>
                          <span>{report.type}</span>
                          <span>•</span>
                          <span>{report.frequency}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(report.status)}`}>
                        {report.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 text-xs mb-3">
                    <div>
                      <p className="text-gray-400">Last Run</p>
                      <p className="font-bold">{report.lastRun}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Executions</p>
                      <p className="font-bold">{report.metadata.executions}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Author</p>
                      <p className="font-bold">{report.author.split('@')[0]}</p>
                    </div>
                  </div>
                  
                  {report.nextRun && (
                    <div className="mb-3 p-2 bg-gray-800 rounded text-xs">
                      <Clock className="w-3 h-3 inline mr-1" />
                      Next run: {report.nextRun}
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {report.metadata.tags.map(tag => (
                        <span key={tag} className="px-2 py-1 bg-gray-800 rounded text-xs">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Play className="w-4 h-4 text-gray-500" />
                      </button>
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Edit className="w-4 h-4 text-gray-500" />
                      </button>
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Copy className="w-4 h-4 text-gray-500" />
                      </button>
                      <button className="p-1 hover:bg-gray-800 rounded">
                        <Share className="w-4 h-4 text-gray-500" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'templates' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Report Templates</h3>
              <p className="text-sm text-gray-400">Get started quickly with pre-built report templates</p>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              {templates.map(template => (
                <div key={template.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="mb-3">
                    <div className="w-full h-24 bg-gray-800 rounded mb-3 flex items-center justify-center">
                      <FileText className="w-8 h-8 text-gray-500" />
                    </div>
                    <h4 className="text-sm font-bold mb-1">{template.name}</h4>
                    <p className="text-xs text-gray-400 mb-2">{template.description}</p>
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="px-2 py-1 bg-blue-900/20 text-blue-500 rounded text-xs">
                        {template.complexity.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">~{template.estimatedTime}</span>
                    </div>
                  </div>
                  
                  <div className="mb-3">
                    <p className="text-xs text-gray-400 mb-1">Includes:</p>
                    <div className="space-y-1">
                      {template.includes.slice(0, 3).map((item, idx) => (
                        <div key={idx} className="flex items-center space-x-1 text-xs">
                          <CheckCircle className="w-3 h-3 text-green-500" />
                          <span>{item}</span>
                        </div>
                      ))}
                      {template.includes.length > 3 && (
                        <span className="text-xs text-gray-500">+{template.includes.length - 3} more</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button className="flex-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                      Use Template
                    </button>
                    <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                      Preview
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'executions' && (
          <>
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-3">Report Executions</h3>
              <p className="text-sm text-gray-400">Track report generation status and download results</p>
            </div>
            
            <div className="space-y-3">
              {executions.map(execution => (
                <div key={execution.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 text-xs rounded ${getStatusColor(execution.status)}`}>
                        {execution.status.toUpperCase()}
                      </span>
                      <div>
                        <h4 className="text-sm font-bold">{execution.reportName}</h4>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>Started: {new Date(execution.startTime).toLocaleString()}</span>
                          {execution.duration && <span>Duration: {execution.duration}s</span>}
                          <span>By: {execution.triggeredBy}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {execution.output && (
                        <div className="text-right text-xs">
                          <p className="text-gray-400">{execution.output.format.toUpperCase()}</p>
                          <p className="text-gray-500">{execution.output.size}MB</p>
                        </div>
                      )}
                      
                      {execution.status === 'completed' && execution.output && (
                        <button className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-xs flex items-center space-x-1">
                          <Download className="w-3 h-3" />
                          <span>Download</span>
                        </button>
                      )}
                      
                      {execution.status === 'running' && (
                        <div className="flex items-center space-x-2">
                          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                          <span className="text-xs text-blue-500">Running...</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {execution.error && (
                    <div className="mt-3 p-2 bg-red-900/20 border border-red-800 rounded text-xs text-red-500">
                      <XCircle className="w-3 h-3 inline mr-1" />
                      {execution.error}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {viewMode === 'builder' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="text-center py-12">
              <Settings className="w-12 h-12 text-gray-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold mb-2">Report Builder</h3>
              <p className="text-sm text-gray-400 mb-6">
                Visual report builder interface would be implemented here
              </p>
              <div className="flex justify-center space-x-3">
                <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm">
                  Start Building
                </button>
                <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm">
                  Import Template
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}