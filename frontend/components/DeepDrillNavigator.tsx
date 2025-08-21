'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  ChevronRight, Search, Filter, Grid, List,
  User, Shield, AlertTriangle, DollarSign,
  Activity, Network, GitBranch, TrendingUp,
  Clock, BarChart3, Eye, Maximize2, Database,
  FileText, Lock, Zap, Target, Layers,
  ArrowRight, ExternalLink, Info
} from 'lucide-react';
import { motion } from 'framer-motion';

interface DeepDrillEntry {
  id: string;
  type: 'rbac' | 'compliance' | 'cost' | 'security' | 'resource' | 'network' | 'prediction';
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  metrics: {
    label: string;
    value: string | number;
    trend?: 'up' | 'down' | 'stable';
    severity?: 'critical' | 'high' | 'medium' | 'low';
  }[];
  drillPath: string;
  hasDeepDrill: boolean;
  relatedItems?: string[];
  lastUpdated?: string;
}

interface DeepDrillNavigatorProps {
  entries: DeepDrillEntry[];
  viewMode?: 'grid' | 'list';
  onDrillDown?: (entry: DeepDrillEntry) => void;
}

export default function DeepDrillNavigator({ 
  entries, 
  viewMode = 'grid',
  onDrillDown 
}: DeepDrillNavigatorProps) {
  const router = useRouter();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [hoveredEntry, setHoveredEntry] = useState<string | null>(null);
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);

  const handleDrillDown = (entry: DeepDrillEntry) => {
    if (onDrillDown) {
      onDrillDown(entry);
    } else {
      // Navigate to the deep-drill view
      router.push(entry.drillPath);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'rbac': return 'bg-purple-100 text-purple-600 border-purple-200';
      case 'compliance': return 'bg-yellow-100 text-yellow-600 border-yellow-200';
      case 'cost': return 'bg-green-100 text-green-600 border-green-200';
      case 'security': return 'bg-red-100 text-red-600 border-red-200';
      case 'resource': return 'bg-blue-100 text-blue-600 border-blue-200';
      case 'network': return 'bg-indigo-100 text-indigo-600 border-indigo-200';
      case 'prediction': return 'bg-pink-100 text-pink-600 border-pink-200';
      default: return 'bg-gray-100 text-gray-600 border-gray-200';
    }
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600';
      case 'high': return 'text-orange-600';
      case 'medium': return 'text-yellow-600';
      case 'low': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const getTrendIcon = (trend?: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-3 h-3 text-red-500" />;
      case 'down': return <TrendingUp className="w-3 h-3 text-green-500 rotate-180" />;
      default: return <Activity className="w-3 h-3 text-gray-400" />;
    }
  };

  const filteredEntries = entries.filter(entry => {
    const matchesSearch = entry.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          entry.subtitle.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = selectedType === 'all' || entry.type === selectedType;
    return matchesSearch && matchesType;
  });

  const renderGridView = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {filteredEntries.map((entry) => (
        <motion.div
          key={entry.id}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          whileHover={{ scale: 1.02 }}
          onHoverStart={() => setHoveredEntry(entry.id)}
          onHoverEnd={() => setHoveredEntry(null)}
          className={`relative bg-white rounded-lg border-2 p-4 cursor-pointer transition-all ${
            entry.hasDeepDrill 
              ? 'hover:shadow-lg border-gray-200 hover:border-blue-300' 
              : 'border-gray-100 opacity-90'
          }`}
          onClick={() => entry.hasDeepDrill && handleDrillDown(entry)}
        >
          {/* Deep Drill Indicator */}
          {entry.hasDeepDrill && (
            <div className="absolute top-2 right-2">
              <div className="relative group">
                <Maximize2 className="w-4 h-4 text-blue-500" />
                <div className="absolute right-0 top-6 hidden group-hover:block z-10">
                  <div className="bg-gray-900 text-white text-xs rounded py-1 px-2 whitespace-nowrap">
                    Click for deep-drill analysis
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Type Badge */}
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mb-3 ${getTypeColor(entry.type)}`}>
            {entry.type.toUpperCase()}
          </div>

          {/* Icon and Title */}
          <div className="flex items-start space-x-3 mb-3">
            <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
              {entry.icon}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-gray-900 text-sm truncate">{entry.title}</h3>
              <p className="text-xs text-gray-600 mt-0.5 line-clamp-2">{entry.subtitle}</p>
            </div>
          </div>

          {/* Metrics */}
          <div className="space-y-2">
            {entry.metrics.slice(0, 3).map((metric, index) => (
              <div key={index} className="flex justify-between items-center text-xs">
                <span className="text-gray-500">{metric.label}:</span>
                <div className="flex items-center space-x-1">
                  <span className={`font-medium ${getSeverityColor(metric.severity)}`}>
                    {metric.value}
                  </span>
                  {metric.trend && getTrendIcon(metric.trend)}
                </div>
              </div>
            ))}
          </div>

          {/* Hover Actions */}
          {hoveredEntry === entry.id && entry.hasDeepDrill && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-blue-50 to-transparent p-4 rounded-b-lg"
            >
              <button type="button" className="w-full px-3 py-1.5 bg-blue-600 text-white text-xs rounded-md hover:bg-blue-700 flex items-center justify-center">
                Drill Down
                <ChevronRight className="w-3 h-3 ml-1" />
              </button>
            </motion.div>
          )}
        </motion.div>
      ))}
    </div>
  );

  const renderListView = () => (
    <div className="space-y-2">
      {filteredEntries.map((entry) => (
        <motion.div
          key={entry.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className={`bg-white rounded-lg border p-4 cursor-pointer transition-all ${
            entry.hasDeepDrill 
              ? 'hover:shadow-md border-gray-200 hover:border-blue-300' 
              : 'border-gray-100 opacity-90'
          }`}
          onClick={() => entry.hasDeepDrill && handleDrillDown(entry)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* Icon */}
              <div className="flex-shrink-0 w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                {entry.icon}
              </div>

              {/* Content */}
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <h3 className="font-semibold text-gray-900">{entry.title}</h3>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getTypeColor(entry.type)}`}>
                    {entry.type}
                  </span>
                </div>
                <p className="text-sm text-gray-600">{entry.subtitle}</p>
              </div>

              {/* Metrics */}
              <div className="flex space-x-6">
                {entry.metrics.slice(0, 4).map((metric, index) => (
                  <div key={index} className="text-center">
                    <div className={`text-lg font-semibold ${getSeverityColor(metric.severity)}`}>
                      {metric.value}
                    </div>
                    <div className="text-xs text-gray-500">{metric.label}</div>
                  </div>
                ))}
              </div>

              {/* Deep Drill Indicator */}
              {entry.hasDeepDrill && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <span className="text-sm font-medium">Analyze</span>
                  <ChevronRight className="w-5 h-5" />
                </div>
              )}
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-between">
        <div className="flex-1 min-w-[300px] max-w-md">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search for items to drill into..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Type Filter */}
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Types</option>
            <option value="rbac">RBAC</option>
            <option value="compliance">Compliance</option>
            <option value="cost">Cost</option>
            <option value="security">Security</option>
            <option value="resource">Resources</option>
            <option value="network">Network</option>
            <option value="prediction">Predictions</option>
          </select>

          {/* View Mode Toggle */}
          <div className="flex bg-gray-100 rounded-md p-1">
            <button type="button"
              onClick={() => setCurrentViewMode('grid')}
              className={`p-2 rounded ${
                currentViewMode === 'grid' 
                  ? 'bg-white shadow-sm text-blue-600' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button type="button"
              onClick={() => setCurrentViewMode('list')}
              className={`p-2 rounded ${
                currentViewMode === 'list' 
                  ? 'bg-white shadow-sm text-blue-600' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-start">
        <Info className="w-5 h-5 text-blue-600 mr-2 mt-0.5" />
        <div className="text-sm text-blue-800">
          <span className="font-medium">Deep-Drill Navigation:</span> Click on any item with the{' '}
          <Maximize2 className="w-3 h-3 inline" /> icon to access detailed analysis. 
          Navigate through multiple levels to uncover root causes, patterns, and actionable insights.
        </div>
      </div>

      {/* Results Count */}
      <div className="text-sm text-gray-600">
        Showing {filteredEntries.length} of {entries.length} items
        {filteredEntries.filter(e => e.hasDeepDrill).length > 0 && (
          <span className="ml-2 text-blue-600">
            ({filteredEntries.filter(e => e.hasDeepDrill).length} with deep-drill)
          </span>
        )}
      </div>

      {/* Content */}
      {currentViewMode === 'grid' ? renderGridView() : renderListView()}

      {/* Empty State */}
      {filteredEntries.length === 0 && (
        <div className="text-center py-12">
          <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No items found</h3>
          <p className="text-gray-600">Try adjusting your search or filter criteria</p>
        </div>
      )}
    </div>
  );
}

// Export sample entries for testing
export const sampleDeepDrillEntries: DeepDrillEntry[] = [
  {
    id: 'rbac-1',
    type: 'rbac',
    title: 'John Doe - Over-provisioned',
    subtitle: '45 unused permissions detected in the last 90 days',
    icon: <User className="w-5 h-5 text-purple-600" />,
    metrics: [
      { label: 'Risk Score', value: '78%', severity: 'high' },
      { label: 'Permissions', value: 127, trend: 'up' },
      { label: 'Last Review', value: '30d ago' }
    ],
    drillPath: '/rbac/users/john-doe/deep-drill',
    hasDeepDrill: true,
    relatedItems: ['role-admin', 'group-it'],
    lastUpdated: '2024-01-15T10:30:00Z'
  },
  {
    id: 'compliance-1',
    type: 'compliance',
    title: 'PCI-DSS Violation',
    subtitle: 'Encryption at rest not enabled for 12 storage accounts',
    icon: <AlertTriangle className="w-5 h-5 text-yellow-600" />,
    metrics: [
      { label: 'Severity', value: 'Critical', severity: 'critical' },
      { label: 'Resources', value: 12 },
      { label: 'Days Open', value: 5 }
    ],
    drillPath: '/compliance/violations/pci-dss-001/deep-drill',
    hasDeepDrill: true,
    lastUpdated: '2024-01-15T09:15:00Z'
  },
  {
    id: 'cost-1',
    type: 'cost',
    title: 'Unusual Spike Detected',
    subtitle: 'Compute costs increased by 235% in the last 24 hours',
    icon: <DollarSign className="w-5 h-5 text-green-600" />,
    metrics: [
      { label: 'Impact', value: '$8,500', severity: 'high' },
      { label: 'Change', value: '+235%', trend: 'up' },
      { label: 'Resources', value: 23 }
    ],
    drillPath: '/cost/anomalies/spike-001/deep-drill',
    hasDeepDrill: true,
    lastUpdated: '2024-01-15T11:00:00Z'
  },
  {
    id: 'security-1',
    type: 'security',
    title: 'Suspicious Login Activity',
    subtitle: 'Multiple failed login attempts from unusual location',
    icon: <Shield className="w-5 h-5 text-red-600" />,
    metrics: [
      { label: 'Risk', value: 'High', severity: 'high' },
      { label: 'Attempts', value: 47 },
      { label: 'Accounts', value: 3 }
    ],
    drillPath: '/security/threats/login-001/deep-drill',
    hasDeepDrill: true,
    lastUpdated: '2024-01-15T10:45:00Z'
  },
  {
    id: 'resource-1',
    type: 'resource',
    title: 'VM Performance Degradation',
    subtitle: 'Production VMs showing 40% performance drop',
    icon: <Database className="w-5 h-5 text-blue-600" />,
    metrics: [
      { label: 'Health', value: 'Degraded', severity: 'medium' },
      { label: 'CPU', value: '92%', trend: 'up' },
      { label: 'Affected', value: 5 }
    ],
    drillPath: '/resources/vms/prod-cluster/deep-drill',
    hasDeepDrill: true,
    lastUpdated: '2024-01-15T10:00:00Z'
  },
  {
    id: 'network-1',
    type: 'network',
    title: 'Unusual Traffic Pattern',
    subtitle: 'Outbound traffic to unknown IPs detected',
    icon: <Network className="w-5 h-5 text-indigo-600" />,
    metrics: [
      { label: 'Severity', value: 'Medium', severity: 'medium' },
      { label: 'Data', value: '2.3 GB' },
      { label: 'IPs', value: 8 }
    ],
    drillPath: '/network/anomalies/traffic-001/deep-drill',
    hasDeepDrill: true,
    lastUpdated: '2024-01-15T09:30:00Z'
  },
  {
    id: 'prediction-1',
    type: 'prediction',
    title: 'Compliance Drift Predicted',
    subtitle: 'ML model predicts 85% chance of compliance violation in 7 days',
    icon: <TrendingUp className="w-5 h-5 text-pink-600" />,
    metrics: [
      { label: 'Confidence', value: '85%', severity: 'high' },
      { label: 'Time', value: '7 days' },
      { label: 'Resources', value: 15 }
    ],
    drillPath: '/predictions/compliance-drift-001/deep-drill',
    hasDeepDrill: true,
    lastUpdated: '2024-01-15T08:00:00Z'
  }
];