'use client';

import React, { useState, useEffect } from 'react';
import {
  TrendingUp, TrendingDown, DollarSign, AlertTriangle,
  ChevronRight, ChevronDown, Calendar, Clock, BarChart3,
  PieChart, Activity, Zap, Database, Server, Cloud,
  ArrowUp, ArrowDown, Minus, Info, Download, RefreshCw,
  Filter, Search, Settings, Target, Lightbulb, GitBranch
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow, subDays, startOfMonth, endOfMonth } from 'date-fns';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart as RePieChart, Pie, Cell,
  Scatter, ScatterChart, ZAxis
} from 'recharts';

interface CostAnomaly {
  anomalyId: string;
  detectedAt: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'investigating' | 'resolved' | 'false_positive';
  type: 'spike' | 'unusual_pattern' | 'new_service' | 'threshold_breach' | 'forecast_deviation';
  title: string;
  description: string;
  impactedAmount: number;
  percentageChange: number;
  affectedResources: AffectedResource[];
  costBreakdown: CostBreakdown;
  usagePatterns: UsagePattern[];
  rootCause: RootCauseAnalysis;
  recommendations: OptimizationRecommendation[];
  historicalContext: HistoricalContext;
  forecastImpact: ForecastImpact;
  similarAnomalies: SimilarAnomaly[];
}

interface AffectedResource {
  resourceId: string;
  resourceName: string;
  resourceType: string;
  resourceGroup: string;
  subscription: string;
  region: string;
  tags: Record<string, string>;
  currentCost: number;
  previousCost: number;
  costIncrease: number;
  percentageIncrease: number;
  usageMetrics: UsageMetrics;
  pricingTier: string;
  resourceHealth: 'healthy' | 'degraded' | 'unhealthy';
  lastModified: string;
  owner: string;
  department: string;
}

interface CostBreakdown {
  byService: ServiceCost[];
  byResource: ResourceCost[];
  byRegion: RegionCost[];
  byTag: TagCost[];
  byTimeOfDay: TimeOfDayCost[];
  byOperation: OperationCost[];
}

interface UsagePattern {
  metric: string;
  unit: string;
  current: number;
  previous: number;
  change: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  anomalyScore: number;
  peakTimes: string[];
  averageDaily: number;
  forecast: number[];
}

interface OptimizationRecommendation {
  recommendationId: string;
  priority: 'immediate' | 'high' | 'medium' | 'low';
  type: 'resize' | 'shutdown' | 'reserved_instance' | 'spot_instance' | 'auto_scaling' | 'tag_resources';
  title: string;
  description: string;
  estimatedSavings: number;
  estimatedSavingsPercentage: number;
  implementationEffort: 'low' | 'medium' | 'high';
  automationAvailable: boolean;
  impactedResources: string[];
  implementationSteps: ImplementationStep[];
  risks: Risk[];
  successMetrics: string[];
}

interface DrillLevel {
  level: number;
  type: 'anomaly' | 'resource' | 'usage' | 'optimization';
  id: string;
  name: string;
  data?: any;
}

export default function CostAnomalyDeepDrill() {
  const [selectedAnomaly, setSelectedAnomaly] = useState<CostAnomaly | null>(null);
  const [selectedResource, setSelectedResource] = useState<AffectedResource | null>(null);
  const [drillPath, setDrillPath] = useState<DrillLevel[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'resources' | 'patterns' | 'recommendations' | 'forecast'>('overview');
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | '90d'>('30d');
  const [comparisonMode, setComparisonMode] = useState<'previous_period' | 'budget' | 'forecast'>('previous_period');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['breakdown']));
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area'>('area');

  const fetchAnomalyDetails = async (anomalyId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/cost/anomalies/${anomalyId}/deep-drill`);
      const data = await response.json();
      setSelectedAnomaly(data);
      setDrillPath([
        { level: 0, type: 'anomaly', id: anomalyId, name: data.title, data }
      ]);
    } catch (error) {
      console.error('Error fetching anomaly details:', error);
    } finally {
      setLoading(false);
    }
  };

  const drillIntoResource = (resource: AffectedResource) => {
    setSelectedResource(resource);
    setDrillPath([
      ...drillPath,
      { level: drillPath.length, type: 'resource', id: resource.resourceId, name: resource.resourceName, data: resource }
    ]);
  };

  const navigateToDrillLevel = (level: number) => {
    const newPath = drillPath.slice(0, level + 1);
    setDrillPath(newPath);
    
    if (level === 0) {
      setSelectedResource(null);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: amount < 10 ? 2 : 0
    }).format(amount);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getChangeIndicator = (change: number) => {
    if (change > 0) return { icon: <ArrowUp className="w-4 h-4" />, color: 'text-red-600' };
    if (change < 0) return { icon: <ArrowDown className="w-4 h-4" />, color: 'text-green-600' };
    return { icon: <Minus className="w-4 h-4" />, color: 'text-gray-600' };
  };

  const renderBreadcrumb = () => (
    <div className="flex items-center space-x-2 text-sm mb-6 p-3 bg-gray-50 rounded-lg">
      <button
        onClick={() => {
          setDrillPath([]);
          setSelectedAnomaly(null);
          setSelectedResource(null);
        }}
        className="text-blue-600 hover:text-blue-800 font-medium"
      >
        Cost Management
      </button>
      {drillPath.map((path, index) => (
        <React.Fragment key={`${path.type}-${path.id}`}>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <button
            onClick={() => navigateToDrillLevel(index)}
            className={`hover:text-blue-800 font-medium ${
              index === drillPath.length - 1 ? 'text-gray-900' : 'text-blue-600'
            }`}
          >
            {path.name}
          </button>
        </React.Fragment>
      ))}
    </div>
  );

  const renderAnomalyOverview = () => {
    if (!selectedAnomaly) return null;

    // Mock data for charts
    const costTrendData = Array.from({ length: 30 }, (_, i) => ({
      date: format(subDays(new Date(), 29 - i), 'MMM dd'),
      actual: Math.random() * 1000 + 500 + (i > 25 ? 800 : 0),
      forecast: Math.random() * 1000 + 500,
      budget: 800
    }));

    const serviceBreakdownData = [
      { name: 'Compute', value: 4500, percentage: 45 },
      { name: 'Storage', value: 2300, percentage: 23 },
      { name: 'Network', value: 1200, percentage: 12 },
      { name: 'Database', value: 1000, percentage: 10 },
      { name: 'Other', value: 1000, percentage: 10 }
    ];

    const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

    return (
      <div className="space-y-6">
        {/* Anomaly Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <div className="flex items-center space-x-3 mb-2">
                <TrendingUp className="w-6 h-6 text-red-600" />
                <h2 className="text-xl font-bold text-gray-900">{selectedAnomaly.title}</h2>
                <span className={`px-2 py-1 text-xs rounded-full ${getSeverityColor(selectedAnomaly.severity)}`}>
                  {selectedAnomaly.severity.toUpperCase()}
                </span>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  selectedAnomaly.status === 'active' ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                }`}>
                  {selectedAnomaly.status.toUpperCase()}
                </span>
              </div>
              <p className="text-gray-600">{selectedAnomaly.description}</p>
            </div>
            <div className="flex space-x-2">
              <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center">
                <Lightbulb className="w-4 h-4 mr-2" />
                View Recommendations
              </button>
              <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200">
                <Download className="w-4 h-4 inline mr-2" />
                Export Analysis
              </button>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-red-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-red-600">
                {formatCurrency(selectedAnomaly.impactedAmount)}
              </div>
              <div className="text-xs text-gray-600">Cost Impact</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-orange-600 flex items-center">
                {getChangeIndicator(selectedAnomaly.percentageChange).icon}
                {Math.abs(selectedAnomaly.percentageChange)}%
              </div>
              <div className="text-xs text-gray-600">Change</div>
            </div>
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-blue-600">
                {selectedAnomaly.affectedResources.length}
              </div>
              <div className="text-xs text-gray-600">Resources</div>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-green-600">
                {formatCurrency(
                  selectedAnomaly.recommendations.reduce((sum, r) => sum + r.estimatedSavings, 0)
                )}
              </div>
              <div className="text-xs text-gray-600">Potential Savings</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-3">
              <div className="text-sm font-bold text-purple-600">
                {formatDistanceToNow(new Date(selectedAnomaly.detectedAt), { addSuffix: true })}
              </div>
              <div className="text-xs text-gray-600">Detected</div>
            </div>
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex justify-between items-center">
            <div className="flex space-x-2">
              {(['24h', '7d', '30d', '90d'] as const).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-4 py-2 rounded-md text-sm font-medium ${
                    timeRange === range
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {range === '24h' ? 'Last 24 Hours' : `Last ${range}`}
                </button>
              ))}
            </div>
            <div className="flex space-x-2">
              {(['line', 'bar', 'area'] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => setChartType(type)}
                  className={`p-2 rounded-md ${
                    chartType === type
                      ? 'bg-blue-100 text-blue-600'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              {['overview', 'resources', 'patterns', 'recommendations', 'forecast'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Cost Trend Chart */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Trend Analysis</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={costTrendData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis tickFormatter={(value) => `$${value}`} />
                      <Tooltip formatter={(value: any) => formatCurrency(value)} />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="actual"
                        stroke="#EF4444"
                        fill="#FEE2E2"
                        strokeWidth={2}
                        name="Actual Cost"
                      />
                      <Area
                        type="monotone"
                        dataKey="forecast"
                        stroke="#3B82F6"
                        fill="#DBEAFE"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        name="Forecast"
                      />
                      <Line
                        type="monotone"
                        dataKey="budget"
                        stroke="#10B981"
                        strokeWidth={2}
                        strokeDasharray="10 5"
                        name="Budget"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Service Breakdown */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost by Service</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <RePieChart>
                        <Pie
                          data={serviceBreakdownData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percentage }) => `${name} ${percentage}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {serviceBreakdownData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value: any) => formatCurrency(value)} />
                      </RePieChart>
                    </ResponsiveContainer>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Cost Drivers</h3>
                    <div className="space-y-3">
                      {serviceBreakdownData.map((service, index) => (
                        <div key={service.name} className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: COLORS[index % COLORS.length] }}
                            />
                            <span className="text-sm font-medium text-gray-900">{service.name}</span>
                          </div>
                          <div className="flex items-center space-x-4">
                            <span className="text-sm text-gray-600">{formatCurrency(service.value)}</span>
                            <div className="w-24 bg-gray-200 rounded-full h-2">
                              <div
                                className="h-2 rounded-full"
                                style={{
                                  width: `${service.percentage}%`,
                                  backgroundColor: COLORS[index % COLORS.length]
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Root Cause Analysis */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Info className="w-5 h-5 text-yellow-600 mr-3 mt-0.5" />
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Root Cause Analysis</h4>
                      <p className="text-sm text-gray-700">
                        {selectedAnomaly.rootCause?.description || 
                         'Significant increase in compute resources usage detected in the production environment. ' +
                         'Analysis indicates auto-scaling triggered due to increased traffic, but scaling policies may need optimization.'}
                      </p>
                      <div className="mt-3 space-y-2">
                        <div className="flex items-center text-sm">
                          <span className="font-medium text-gray-700 mr-2">Primary Factor:</span>
                          <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded">
                            Auto-scaling misconfiguration
                          </span>
                        </div>
                        <div className="flex items-center text-sm">
                          <span className="font-medium text-gray-700 mr-2">Confidence:</span>
                          <span className="text-yellow-700">85%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'resources' && (
              <div className="space-y-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Affected Resources ({selectedAnomaly.affectedResources.length})
                  </h3>
                  <div className="flex space-x-2">
                    <select className="px-3 py-2 border border-gray-300 rounded-md text-sm">
                      <option>Sort by: Cost Impact</option>
                      <option>Sort by: % Change</option>
                      <option>Sort by: Resource Type</option>
                    </select>
                    <button className="px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 text-sm">
                      <Filter className="w-4 h-4 inline mr-1" />
                      Filter
                    </button>
                  </div>
                </div>

                {selectedAnomaly.affectedResources.map((resource) => (
                  <motion.div
                    key={resource.resourceId}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => drillIntoResource(resource)}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <Server className="w-4 h-4 text-gray-400" />
                          <h4 className="font-medium text-gray-900">{resource.resourceName}</h4>
                          <span className="text-xs text-gray-500">({resource.resourceType})</span>
                          <span className={`px-2 py-0.5 text-xs rounded-full ${
                            resource.resourceHealth === 'healthy' 
                              ? 'bg-green-100 text-green-600' 
                              : resource.resourceHealth === 'degraded'
                              ? 'bg-yellow-100 text-yellow-600'
                              : 'bg-red-100 text-red-600'
                          }`}>
                            {resource.resourceHealth}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Current Cost:</span>
                            <span className="ml-1 font-medium text-red-600">
                              {formatCurrency(resource.currentCost)}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">Previous:</span>
                            <span className="ml-1 font-medium">
                              {formatCurrency(resource.previousCost)}
                            </span>
                          </div>
                          <div className="flex items-center">
                            <span className="text-gray-500">Change:</span>
                            <span className={`ml-1 flex items-center font-medium ${
                              resource.percentageIncrease > 0 ? 'text-red-600' : 'text-green-600'
                            }`}>
                              {getChangeIndicator(resource.percentageIncrease).icon}
                              {Math.abs(resource.percentageIncrease)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">Region:</span>
                            <span className="ml-1 font-medium">{resource.region}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Owner:</span>
                            <span className="ml-1 font-medium">{resource.owner}</span>
                          </div>
                        </div>

                        <div className="mt-3 flex items-center space-x-4 text-xs text-gray-600">
                          <span>Pricing Tier: {resource.pricingTier}</span>
                          <span>Department: {resource.department}</span>
                          <span>Last Modified: {formatDistanceToNow(new Date(resource.lastModified), { addSuffix: true })}</span>
                        </div>
                      </div>
                      
                      <ChevronRight className="w-5 h-5 text-gray-400 mt-1" />
                    </div>
                  </motion.div>
                ))}
              </div>
            )}

            {activeTab === 'patterns' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Pattern Analysis</h3>
                
                {selectedAnomaly.usagePatterns?.map((pattern, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h4 className="font-medium text-gray-900">{pattern.metric}</h4>
                        <p className="text-sm text-gray-600">Unit: {pattern.unit}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        pattern.anomalyScore > 0.8 
                          ? 'bg-red-100 text-red-600'
                          : pattern.anomalyScore > 0.5
                          ? 'bg-yellow-100 text-yellow-600'
                          : 'bg-green-100 text-green-600'
                      }`}>
                        Anomaly Score: {(pattern.anomalyScore * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-xs text-gray-600">Current</div>
                        <div className="text-lg font-semibold">{pattern.current.toLocaleString()}</div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-xs text-gray-600">Previous</div>
                        <div className="text-lg font-semibold">{pattern.previous.toLocaleString()}</div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-xs text-gray-600">Change</div>
                        <div className={`text-lg font-semibold flex items-center ${
                          pattern.change > 0 ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {getChangeIndicator(pattern.change).icon}
                          {Math.abs(pattern.change)}%
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <div className="text-xs text-gray-600">Daily Avg</div>
                        <div className="text-lg font-semibold">{pattern.averageDaily.toLocaleString()}</div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      <span>Trend: {pattern.trend}</span>
                      <span>Peak Times: {pattern.peakTimes.join(', ')}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === 'recommendations' && (
              <div className="space-y-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Optimization Recommendations
                  </h3>
                  <div className="text-sm text-gray-600">
                    Total Potential Savings: 
                    <span className="ml-2 text-lg font-semibold text-green-600">
                      {formatCurrency(
                        selectedAnomaly.recommendations.reduce((sum, r) => sum + r.estimatedSavings, 0)
                      )}
                    </span>
                  </div>
                </div>

                {selectedAnomaly.recommendations.map((rec, index) => (
                  <motion.div
                    key={rec.recommendationId}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="border border-gray-200 rounded-lg p-4"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-start space-x-3">
                        <Lightbulb className={`w-5 h-5 mt-0.5 ${
                          rec.priority === 'immediate' ? 'text-red-600' :
                          rec.priority === 'high' ? 'text-orange-600' :
                          rec.priority === 'medium' ? 'text-yellow-600' :
                          'text-blue-600'
                        }`} />
                        <div>
                          <h4 className="font-medium text-gray-900">{rec.title}</h4>
                          <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-semibold text-green-600">
                          {formatCurrency(rec.estimatedSavings)}
                        </div>
                        <div className="text-xs text-gray-500">
                          {rec.estimatedSavingsPercentage}% reduction
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4 mb-3">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        rec.priority === 'immediate' ? 'bg-red-100 text-red-600' :
                        rec.priority === 'high' ? 'bg-orange-100 text-orange-600' :
                        rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-600' :
                        'bg-blue-100 text-blue-600'
                      }`}>
                        {rec.priority} priority
                      </span>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        rec.implementationEffort === 'low' ? 'bg-green-100 text-green-600' :
                        rec.implementationEffort === 'medium' ? 'bg-yellow-100 text-yellow-600' :
                        'bg-red-100 text-red-600'
                      }`}>
                        {rec.implementationEffort} effort
                      </span>
                      {rec.automationAvailable && (
                        <span className="px-2 py-1 text-xs bg-purple-100 text-purple-600 rounded-full flex items-center">
                          <Zap className="w-3 h-3 mr-1" />
                          Automation Available
                        </span>
                      )}
                    </div>

                    {rec.implementationSteps && (
                      <details className="mt-3">
                        <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                          Implementation Steps ({rec.implementationSteps.length})
                        </summary>
                        <ol className="mt-2 ml-4 space-y-1 list-decimal list-inside">
                          {rec.implementationSteps.map((step: any, i: number) => (
                            <li key={i} className="text-sm text-gray-600">{step.description || step}</li>
                          ))}
                        </ol>
                      </details>
                    )}

                    <div className="mt-3 flex space-x-2">
                      {rec.automationAvailable && (
                        <button className="px-3 py-1 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700">
                          Auto-Implement
                        </button>
                      )}
                      <button className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-md hover:bg-gray-200">
                        View Details
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}

            {activeTab === 'forecast' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Forecast & Impact</h3>
                
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Target className="w-5 h-5 text-blue-600 mr-3 mt-0.5" />
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Projected Impact</h4>
                      <p className="text-sm text-gray-700">
                        If this anomaly continues unchecked, projected monthly costs will increase by{' '}
                        <span className="font-semibold text-red-600">
                          {formatCurrency(selectedAnomaly.forecastImpact?.monthlyIncrease || 12500)}
                        </span>{' '}
                        representing a <span className="font-semibold">{selectedAnomaly.forecastImpact?.percentageIncrease || 35}%</span> increase
                        from baseline.
                      </p>
                      <div className="mt-3 grid grid-cols-3 gap-4">
                        <div>
                          <div className="text-xs text-gray-600">Next 7 Days</div>
                          <div className="text-lg font-semibold text-red-600">
                            +{formatCurrency(selectedAnomaly.forecastImpact?.weeklyIncrease || 2900)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-600">Next 30 Days</div>
                          <div className="text-lg font-semibold text-red-600">
                            +{formatCurrency(selectedAnomaly.forecastImpact?.monthlyIncrease || 12500)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-600">Quarterly Impact</div>
                          <div className="text-lg font-semibold text-red-600">
                            +{formatCurrency(selectedAnomaly.forecastImpact?.quarterlyIncrease || 37500)}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Budget Impact Analysis</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">Monthly Budget</span>
                      <span className="font-medium">{formatCurrency(50000)}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">Current Spend</span>
                      <span className="font-medium">{formatCurrency(35000)}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-lg">
                      <span className="text-sm text-gray-600">Projected with Anomaly</span>
                      <span className="font-medium text-orange-600">{formatCurrency(47500)}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                      <span className="text-sm text-gray-600">Budget Variance</span>
                      <span className="font-medium text-red-600">-{formatCurrency(2500)} (95% utilized)</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderResourceDetail = () => {
    if (!selectedResource) return null;

    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{selectedResource.resourceName}</h2>
            <p className="text-sm text-gray-600 mt-1">
              {selectedResource.resourceType} • {selectedResource.resourceId}
            </p>
          </div>
          <div className="flex space-x-2">
            <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
              Optimize Resource
            </button>
            <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200">
              View in Azure
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Cost Analysis</h3>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Current Monthly Cost</span>
                <span className="text-sm font-medium text-red-600">{formatCurrency(selectedResource.currentCost)}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Previous Monthly Cost</span>
                <span className="text-sm font-medium">{formatCurrency(selectedResource.previousCost)}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Cost Increase</span>
                <span className="text-sm font-medium text-red-600">
                  +{formatCurrency(selectedResource.costIncrease)} ({selectedResource.percentageIncrease}%)
                </span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Pricing Tier</span>
                <span className="text-sm font-medium">{selectedResource.pricingTier}</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Resource Details</h3>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Resource Group</span>
                <span className="text-sm font-medium">{selectedResource.resourceGroup}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Subscription</span>
                <span className="text-sm font-medium">{selectedResource.subscription}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Region</span>
                <span className="text-sm font-medium">{selectedResource.region}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Owner</span>
                <span className="text-sm font-medium">{selectedResource.owner}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-600">Department</span>
                <span className="text-sm font-medium">{selectedResource.department}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Usage Metrics */}
        <div className="mt-6">
          <h3 className="font-semibold text-gray-900 mb-4">Usage Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-600">CPU Usage</div>
              <div className="text-xl font-semibold">{selectedResource.usageMetrics?.cpu || 78}%</div>
              <div className="text-xs text-red-600">+12% from last period</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-600">Memory</div>
              <div className="text-xl font-semibold">{selectedResource.usageMetrics?.memory || 65}%</div>
              <div className="text-xs text-green-600">-5% from last period</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-600">Storage</div>
              <div className="text-xl font-semibold">{selectedResource.usageMetrics?.storage || 450} GB</div>
              <div className="text-xs text-red-600">+50 GB from last period</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-600">Network</div>
              <div className="text-xl font-semibold">{selectedResource.usageMetrics?.network || 2.3} TB</div>
              <div className="text-xs text-red-600">+0.8 TB from last period</div>
            </div>
          </div>
        </div>

        {/* Tags */}
        <div className="mt-6">
          <h3 className="font-semibold text-gray-900 mb-4">Resource Tags</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(selectedResource.tags).map(([key, value]) => (
              <span key={key} className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                {key}: {value}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Cost Anomaly Deep-Drill Analysis</h1>
          <p className="text-gray-600">Comprehensive cost analysis and optimization recommendations</p>
        </div>

        {/* Breadcrumb */}
        {renderBreadcrumb()}

        {/* Main Content */}
        <AnimatePresence mode="wait">
          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
          ) : selectedResource ? (
            renderResourceDetail()
          ) : selectedAnomaly ? (
            renderAnomalyOverview()
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
              <TrendingUp className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Select a Cost Anomaly</h3>
              <p className="text-gray-600 mb-6">
                Choose an anomaly from the cost management dashboard to begin deep-drill analysis
              </p>
              <button 
                onClick={() => fetchAnomalyDetails('anomaly-123')}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Load Sample Anomaly
              </button>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// Type definitions
interface ServiceCost {
  service: string;
  cost: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
}

interface ResourceCost {
  resourceId: string;
  resourceName: string;
  cost: number;
  percentage: number;
}

interface RegionCost {
  region: string;
  cost: number;
  percentage: number;
}

interface TagCost {
  tagKey: string;
  tagValue: string;
  cost: number;
  percentage: number;
}

interface TimeOfDayCost {
  hour: number;
  cost: number;
  averageCost: number;
}

interface OperationCost {
  operation: string;
  count: number;
  cost: number;
}

interface UsageMetrics {
  cpu?: number;
  memory?: number;
  storage?: number;
  network?: number;
  [key: string]: number | undefined;
}

interface RootCauseAnalysis {
  description: string;
  primaryFactor: string;
  contributingFactors: string[];
  confidence: number;
  evidence: string[];
}

interface HistoricalContext {
  similarIncidents: number;
  averageResolutionTime: string;
  commonCauses: string[];
  seasonalPattern: boolean;
}

interface ForecastImpact {
  weeklyIncrease: number;
  monthlyIncrease: number;
  quarterlyIncrease: number;
  percentageIncrease: number;
  budgetImpact: string;
}

interface SimilarAnomaly {
  anomalyId: string;
  date: string;
  similarity: number;
  resolution: string;
}

interface ImplementationStep {
  stepNumber: number;
  description: string;
  estimatedTime: string;
  requiresDowntime: boolean;
}

interface Risk {
  riskType: string;
  description: string;
  mitigation: string;
  likelihood: 'low' | 'medium' | 'high';
}