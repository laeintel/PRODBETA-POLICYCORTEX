'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Scale, Shield, AlertTriangle, CheckCircle, Clock, 
  Plus, Search, Filter, Download, Upload, RotateCcw,
  Eye, EyeOff, Copy, Trash2, Edit, Settings, 
  Calendar, Map, Activity, Users, Lock, Unlock,
  FileText, Globe, Zap, Network, Server, Target,
  TrendingUp, TrendingDown, BarChart3, PieChart
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Badge } from '../../../components/ui/badge';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../../../components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Progress } from '../../../components/ui/progress';
import { Textarea } from '../../../components/ui/textarea';
import { Label } from '../../../components/ui/label';

interface GovernanceMetrics {
  overallCompliance: number;
  complianceTrend: number;
  activePolicies: number;
  policyViolations: number;
  riskScore: number;
  costOptimization: number;
  resourcesManaged: number;
  governanceActions: number;
}

interface ComplianceArea {
  id: string;
  name: string;
  category: 'security' | 'cost' | 'operational' | 'regulatory';
  complianceScore: number;
  totalPolicies: number;
  activePolicies: number;
  violations: number;
  lastAssessment: string;
  trend: number;
  priority: 'high' | 'medium' | 'low';
}

interface RecentActivity {
  id: string;
  type: 'policy_created' | 'violation_detected' | 'remediation_completed' | 'assessment_completed';
  title: string;
  description: string;
  timestamp: string;
  severity: 'high' | 'medium' | 'low' | 'info';
  impactedResources: number;
}

export default function GovernanceOverviewPage() {
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<string>('7d');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const [metrics, setMetrics] = useState<GovernanceMetrics>({
    overallCompliance: 0,
    complianceTrend: 0,
    activePolicies: 0,
    policyViolations: 0,
    riskScore: 0,
    costOptimization: 0,
    resourcesManaged: 0,
    governanceActions: 0
  });

  const [complianceAreas, setComplianceAreas] = useState<ComplianceArea[]>([]);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);

  // Mock data generation
  useEffect(() => {
    const generateMockData = () => {
      // Generate compliance areas
      const areas: ComplianceArea[] = [
        {
          id: 'security-compliance',
          name: 'Security Compliance',
          category: 'security',
          complianceScore: 87,
          totalPolicies: 45,
          activePolicies: 42,
          violations: 8,
          lastAssessment: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
          trend: 5,
          priority: 'high'
        },
        {
          id: 'cost-governance',
          name: 'Cost Governance',
          category: 'cost',
          complianceScore: 73,
          totalPolicies: 28,
          activePolicies: 25,
          violations: 12,
          lastAssessment: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
          trend: -3,
          priority: 'high'
        },
        {
          id: 'operational-standards',
          name: 'Operational Standards',
          category: 'operational',
          complianceScore: 91,
          totalPolicies: 35,
          activePolicies: 34,
          violations: 3,
          lastAssessment: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
          trend: 8,
          priority: 'medium'
        },
        {
          id: 'regulatory-compliance',
          name: 'Regulatory Compliance',
          category: 'regulatory',
          complianceScore: 96,
          totalPolicies: 52,
          activePolicies: 51,
          violations: 2,
          lastAssessment: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
          trend: 2,
          priority: 'high'
        },
        {
          id: 'data-governance',
          name: 'Data Governance',
          category: 'security',
          complianceScore: 82,
          totalPolicies: 22,
          activePolicies: 20,
          violations: 5,
          lastAssessment: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
          trend: 1,
          priority: 'medium'
        },
        {
          id: 'resource-optimization',
          name: 'Resource Optimization',
          category: 'cost',
          complianceScore: 78,
          totalPolicies: 18,
          activePolicies: 16,
          violations: 7,
          lastAssessment: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          trend: -1,
          priority: 'low'
        }
      ];

      // Generate recent activity
      const activities: RecentActivity[] = [
        {
          id: 'activity-1',
          type: 'violation_detected',
          title: 'Cost Budget Exceeded',
          description: 'Virtual machines in East US exceeded monthly budget by 15%',
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          severity: 'high',
          impactedResources: 23
        },
        {
          id: 'activity-2',
          type: 'policy_created',
          title: 'New Security Policy Created',
          description: 'Mandatory encryption policy for storage accounts deployed',
          timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
          severity: 'info',
          impactedResources: 145
        },
        {
          id: 'activity-3',
          type: 'remediation_completed',
          title: 'Security Vulnerability Remediated',
          description: 'Unencrypted storage accounts have been secured automatically',
          timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
          severity: 'medium',
          impactedResources: 8
        },
        {
          id: 'activity-4',
          type: 'assessment_completed',
          title: 'Compliance Assessment Completed',
          description: 'Quarterly security compliance assessment finished with 87% score',
          timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
          severity: 'info',
          impactedResources: 342
        },
        {
          id: 'activity-5',
          type: 'violation_detected',
          title: 'Tagging Policy Violation',
          description: 'Resources found without required cost center tags',
          timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
          severity: 'medium',
          impactedResources: 15
        }
      ];

      // Calculate overall metrics
      const avgCompliance = areas.reduce((sum, area) => sum + area.complianceScore, 0) / areas.length;
      const totalPolicies = areas.reduce((sum, area) => sum + area.activePolicies, 0);
      const totalViolations = areas.reduce((sum, area) => sum + area.violations, 0);
      const avgTrend = areas.reduce((sum, area) => sum + area.trend, 0) / areas.length;

      const governanceMetrics: GovernanceMetrics = {
        overallCompliance: Math.round(avgCompliance),
        complianceTrend: Math.round(avgTrend),
        activePolicies: totalPolicies,
        policyViolations: totalViolations,
        riskScore: Math.max(0, 100 - avgCompliance),
        costOptimization: 82,
        resourcesManaged: 1247,
        governanceActions: 156
      };

      return { areas, activities, governanceMetrics };
    };

    setTimeout(() => {
      const { areas, activities, governanceMetrics } = generateMockData();
      setComplianceAreas(areas);
      setRecentActivity(activities);
      setMetrics(governanceMetrics);
      setLoading(false);
    }, 1000);
  }, [timeRange]);

  const filteredAreas = complianceAreas.filter(area => 
    selectedCategory === 'all' || area.category === selectedCategory
  );

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'security': return 'text-red-400 bg-red-500/10 border-red-500/20';
      case 'cost': return 'text-green-400 bg-green-500/10 border-green-500/20';
      case 'operational': return 'text-blue-400 bg-blue-500/10 border-blue-500/20';
      case 'regulatory': return 'text-purple-400 bg-purple-500/10 border-purple-500/20';
      default: return 'text-gray-400 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-blue-400';
      case 'info': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'policy_created': return <Plus className="h-4 w-4" />;
      case 'violation_detected': return <AlertTriangle className="h-4 w-4" />;
      case 'remediation_completed': return <CheckCircle className="h-4 w-4" />;
      case 'assessment_completed': return <BarChart3 className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white p-6">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-800 rounded w-1/4 mb-6"></div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-32 bg-gray-800 rounded"></div>
              ))}
            </div>
            <div className="h-96 bg-gray-800 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-500/10 rounded-lg">
              <Scale className="h-8 w-8 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Governance Overview</h1>
              <p className="text-gray-400">Enterprise cloud governance and compliance dashboard</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-[120px] bg-gray-900 border-gray-700">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="24h">Last 24h</SelectItem>
                <SelectItem value="7d">Last 7 days</SelectItem>
                <SelectItem value="30d">Last 30 days</SelectItem>
                <SelectItem value="90d">Last 90 days</SelectItem>
              </SelectContent>
            </Select>
            <Button className="bg-indigo-600 hover:bg-indigo-700">
              <Download className="h-4 w-4 mr-2" />
              Export Report
            </Button>
          </div>
        </motion.div>

        {/* Key Metrics Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Overall Compliance</p>
                  <p className="text-2xl font-bold text-white">{metrics.overallCompliance}%</p>
                </div>
                <Scale className="h-8 w-8 text-indigo-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                {metrics.complianceTrend > 0 ? (
                  <TrendingUp className="h-4 w-4 text-green-400" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-400" />
                )}
                <span className={`text-sm ${metrics.complianceTrend > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {metrics.complianceTrend > 0 ? '+' : ''}{metrics.complianceTrend}% this period
                </span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Active Policies</p>
                  <p className="text-2xl font-bold text-white">{metrics.activePolicies}</p>
                </div>
                <FileText className="h-8 w-8 text-blue-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-400" />
                <span className="text-sm text-green-400">Actively Enforced</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Policy Violations</p>
                  <p className="text-2xl font-bold text-red-400">{metrics.policyViolations}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Clock className="h-4 w-4 text-yellow-400" />
                <span className="text-sm text-yellow-400">Require Attention</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Resources Managed</p>
                  <p className="text-2xl font-bold text-white">{metrics.resourcesManaged.toLocaleString()}</p>
                </div>
                <Server className="h-8 w-8 text-green-400" />
              </div>
              <div className="mt-2 flex items-center space-x-2">
                <Activity className="h-4 w-4 text-green-400" />
                <span className="text-sm text-green-400">Under Governance</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Compliance Areas */}
          <div className="lg:col-span-2 space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Compliance Areas</h2>
                <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                  <SelectTrigger className="w-[180px] bg-gray-900 border-gray-700">
                    <SelectValue placeholder="Filter by category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    <SelectItem value="security">Security</SelectItem>
                    <SelectItem value="cost">Cost</SelectItem>
                    <SelectItem value="operational">Operational</SelectItem>
                    <SelectItem value="regulatory">Regulatory</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <AnimatePresence>
                  {filteredAreas.map((area, index) => (
                    <motion.div
                      key={area.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <Card className="bg-gray-900/50 border-gray-800">
                        <CardContent className="p-6">
                          <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center space-x-3">
                              <div>
                                <h3 className="font-semibold text-white">{area.name}</h3>
                                <div className="flex items-center space-x-2 mt-1">
                                  <Badge className={getCategoryColor(area.category)}>
                                    {area.category}
                                  </Badge>
                                  <span className={`text-sm ${getPriorityColor(area.priority)}`}>
                                    {area.priority} priority
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-2xl font-bold text-white">{area.complianceScore}%</div>
                              <div className="flex items-center space-x-1">
                                {area.trend > 0 ? (
                                  <TrendingUp className="h-4 w-4 text-green-400" />
                                ) : area.trend < 0 ? (
                                  <TrendingDown className="h-4 w-4 text-red-400" />
                                ) : null}
                                <span className={`text-sm ${area.trend > 0 ? 'text-green-400' : area.trend < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                                  {area.trend !== 0 && (area.trend > 0 ? '+' : '')}{area.trend}%
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="mb-4">
                            <Progress value={area.complianceScore} className="h-2" />
                          </div>

                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-gray-400">Active Policies</p>
                              <p className="text-white font-medium">{area.activePolicies}/{area.totalPolicies}</p>
                            </div>
                            <div>
                              <p className="text-gray-400">Violations</p>
                              <p className="text-red-400 font-medium">{area.violations}</p>
                            </div>
                            <div>
                              <p className="text-gray-400">Last Assessment</p>
                              <p className="text-white font-medium">
                                {new Date(area.lastAssessment).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>

          {/* Recent Activity */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Activity className="h-5 w-5 text-indigo-400" />
                    <span>Recent Activity</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6 pt-0">
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {recentActivity.map((activity, index) => (
                      <motion.div
                        key={activity.id}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-start space-x-3 p-3 rounded-lg bg-gray-800/30"
                      >
                        <div className={`p-1 rounded ${getSeverityColor(activity.severity)}`}>
                          {getActivityIcon(activity.type)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-white text-sm">{activity.title}</h4>
                          <p className="text-xs text-gray-400 mt-1">{activity.description}</p>
                          <div className="flex items-center space-x-2 mt-2">
                            <span className="text-xs text-gray-500">
                              {new Date(activity.timestamp).toLocaleString()}
                            </span>
                            <span className="text-xs text-gray-500">•</span>
                            <span className="text-xs text-gray-500">
                              {activity.impactedResources} resources
                            </span>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Zap className="h-5 w-5 text-yellow-400" />
                    <span>Quick Actions</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6 pt-0">
                  <div className="space-y-3">
                    <Button className="w-full justify-start bg-indigo-600 hover:bg-indigo-700">
                      <Plus className="h-4 w-4 mr-2" />
                      Create New Policy
                    </Button>
                    <Button variant="outline" className="w-full justify-start border-gray-700">
                      <BarChart3 className="h-4 w-4 mr-2" />
                      Run Compliance Scan
                    </Button>
                    <Button variant="outline" className="w-full justify-start border-gray-700">
                      <AlertTriangle className="h-4 w-4 mr-2" />
                      Review Violations
                    </Button>
                    <Button variant="outline" className="w-full justify-start border-gray-700">
                      <Download className="h-4 w-4 mr-2" />
                      Export Compliance Report
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}