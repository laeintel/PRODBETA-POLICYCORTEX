'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Share2, DollarSign, PieChart, BarChart3, Users, Building2,
  Tags, Filter, Download, Upload, Settings, TrendingUp,
  Calculator, Grid3x3, Layers, AlertCircle, CheckCircle,
  Clock, Calendar, ArrowUpRight, ArrowDownRight, RefreshCw,
  Target, Zap, Shield, Database, Activity, FileText,
  ChevronRight, Play, Pause, Edit, Trash2, Copy, Plus
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../components/ui/select';
import { Badge } from '../../../components/ui/badge';
import { Alert, AlertDescription } from '../../../components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Progress } from '../../../components/ui/progress';
import { Switch } from '../../../components/ui/switch';
import { Checkbox } from '../../../components/ui/checkbox';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar, Doughnut, Pie } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface AllocationRule {
  id: string;
  name: string;
  type: 'percentage' | 'usage' | 'fixed' | 'custom' | 'tag-based';
  source: string;
  targets: string[];
  formula: string;
  status: 'active' | 'inactive' | 'draft' | 'testing';
  accuracy: number;
  lastRun: string;
  nextRun: string;
  totalAllocated: number;
  runsCount: number;
  tags?: string[];
}

interface CostCenter {
  id: string;
  name: string;
  code: string;
  department: string;
  manager: string;
  budget: number;
  allocated: number;
  variance: number;
  trend: 'up' | 'down' | 'stable';
  tags: string[];
  lastMonth: number;
  ytd: number;
}

interface AllocationHistory {
  id: string;
  period: string;
  totalCost: number;
  allocatedCost: number;
  unallocatedCost: number;
  accuracy: number;
  status: 'completed' | 'processing' | 'failed' | 'partial';
  runTime: string;
  duration: number;
  ruleCount: number;
}

interface AllocationInsight {
  id: string;
  type: 'warning' | 'success' | 'info' | 'error';
  title: string;
  description: string;
  impact: string;
  action?: string;
  value?: number;
}

export default function CostAllocationPage() {
  const [rules, setRules] = useState<AllocationRule[]>([]);
  const [costCenters, setCostCenters] = useState<CostCenter[]>([]);
  const [history, setHistory] = useState<AllocationHistory[]>([]);
  const [insights, setInsights] = useState<AllocationInsight[]>([]);
  const [selectedRule, setSelectedRule] = useState<AllocationRule | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState('current');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filterStatus, setFilterStatus] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [autoAllocate, setAutoAllocate] = useState(true);
  const [allocationRunning, setAllocationRunning] = useState(false);
  const [selectedCenters, setSelectedCenters] = useState<string[]>([]);

  // Mock data generation
  useEffect(() => {
    const mockRules: AllocationRule[] = [
      {
        id: 'rule-1',
        name: 'Infrastructure Costs by Usage',
        type: 'usage',
        source: 'Azure Infrastructure',
        targets: ['Engineering', 'Marketing', 'Sales', 'Operations'],
        formula: 'CPU_HOURS * 0.45 + STORAGE_GB * 0.12 + NETWORK_GB * 0.08',
        status: 'active',
        accuracy: 98.5,
        lastRun: '2024-01-19T10:30:00Z',
        nextRun: '2024-01-20T10:30:00Z',
        totalAllocated: 2456789,
        runsCount: 365,
        tags: ['infrastructure', 'compute', 'storage']
      },
      {
        id: 'rule-2',
        name: 'Software Licenses by Headcount',
        type: 'percentage',
        source: 'Software Licenses',
        targets: ['All Departments'],
        formula: 'HEADCOUNT / TOTAL_HEADCOUNT * 100',
        status: 'active',
        accuracy: 100,
        lastRun: '2024-01-19T09:00:00Z',
        nextRun: '2024-01-20T09:00:00Z',
        totalAllocated: 845000,
        runsCount: 365,
        tags: ['licenses', 'software']
      },
      {
        id: 'rule-3',
        name: 'Shared Services Fixed Allocation',
        type: 'fixed',
        source: 'Shared Services',
        targets: ['Operations', 'Finance', 'HR', 'Legal'],
        formula: 'FIXED_AMOUNT / TARGET_COUNT',
        status: 'active',
        accuracy: 100,
        lastRun: '2024-01-19T08:00:00Z',
        nextRun: '2024-01-20T08:00:00Z',
        totalAllocated: 520000,
        runsCount: 365,
        tags: ['shared-services']
      },
      {
        id: 'rule-4',
        name: 'AI/ML Compute by Projects',
        type: 'custom',
        source: 'AI/ML Resources',
        targets: ['Data Science', 'Research', 'Product'],
        formula: 'CUSTOM_ALGORITHM(gpu_hours, model_runs, data_processed)',
        status: 'testing',
        accuracy: 95.2,
        lastRun: '2024-01-18T15:00:00Z',
        nextRun: '2024-01-19T15:00:00Z',
        totalAllocated: 1234567,
        runsCount: 45,
        tags: ['ai', 'ml', 'gpu']
      },
      {
        id: 'rule-5',
        name: 'Tag-based Auto Allocation',
        type: 'tag-based',
        source: 'All Tagged Resources',
        targets: ['Dynamic based on tags'],
        formula: 'AUTO_DETECT(resource_tags)',
        status: 'active',
        accuracy: 97.8,
        lastRun: '2024-01-19T12:00:00Z',
        nextRun: '2024-01-20T00:00:00Z',
        totalAllocated: 3456789,
        runsCount: 180,
        tags: ['auto', 'tag-based']
      }
    ];

    const mockCenters: CostCenter[] = [
      {
        id: 'cc-1',
        name: 'Engineering',
        code: 'ENG-001',
        department: 'Technology',
        manager: 'John Smith',
        budget: 500000,
        allocated: 425000,
        variance: -15,
        trend: 'down',
        tags: ['Azure', 'Development', 'Infrastructure'],
        lastMonth: 380000,
        ytd: 4250000
      },
      {
        id: 'cc-2',
        name: 'Marketing',
        code: 'MKT-001',
        department: 'Marketing',
        manager: 'Sarah Johnson',
        budget: 250000,
        allocated: 275000,
        variance: 10,
        trend: 'up',
        tags: ['SaaS', 'Analytics', 'Campaigns'],
        lastMonth: 260000,
        ytd: 2750000
      },
      {
        id: 'cc-3',
        name: 'Sales',
        code: 'SLS-001',
        department: 'Sales',
        manager: 'Mike Wilson',
        budget: 300000,
        allocated: 285000,
        variance: -5,
        trend: 'stable',
        tags: ['CRM', 'Tools', 'Travel'],
        lastMonth: 290000,
        ytd: 2850000
      },
      {
        id: 'cc-4',
        name: 'Operations',
        code: 'OPS-001',
        department: 'Operations',
        manager: 'Lisa Brown',
        budget: 400000,
        allocated: 380000,
        variance: -5,
        trend: 'down',
        tags: ['Infrastructure', 'Security', 'Monitoring'],
        lastMonth: 395000,
        ytd: 3800000
      },
      {
        id: 'cc-5',
        name: 'Data Science',
        code: 'DS-001',
        department: 'Research',
        manager: 'Dr. Chen Wu',
        budget: 350000,
        allocated: 420000,
        variance: 20,
        trend: 'up',
        tags: ['AI', 'ML', 'GPU', 'Research'],
        lastMonth: 380000,
        ytd: 4200000
      },
      {
        id: 'cc-6',
        name: 'Finance',
        code: 'FIN-001',
        department: 'Finance',
        manager: 'Robert Taylor',
        budget: 150000,
        allocated: 145000,
        variance: -3.3,
        trend: 'stable',
        tags: ['ERP', 'Reporting', 'Compliance'],
        lastMonth: 148000,
        ytd: 1450000
      }
    ];

    const mockHistory: AllocationHistory[] = Array.from({ length: 12 }, (_, i) => ({
      id: `hist-${i + 1}`,
      period: new Date(Date.now() - i * 30 * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
      totalCost: 1500000 + Math.random() * 200000,
      allocatedCost: 1400000 + Math.random() * 180000,
      unallocatedCost: 50000 + Math.random() * 20000,
      accuracy: 95 + Math.random() * 5,
      status: i === 0 ? 'processing' : i % 10 === 0 ? 'partial' : 'completed',
      runTime: new Date(Date.now() - i * 30 * 24 * 60 * 60 * 1000).toISOString(),
      duration: 45 + Math.random() * 120,
      ruleCount: 5 + Math.floor(Math.random() * 3)
    }));

    const mockInsights: AllocationInsight[] = [
      {
        id: 'insight-1',
        type: 'warning',
        title: 'Engineering Cost Overrun',
        description: 'Engineering department has exceeded budget by 15% for 3 consecutive months',
        impact: 'Budget variance of $75,000',
        action: 'Review infrastructure optimization opportunities',
        value: 75000
      },
      {
        id: 'insight-2',
        type: 'success',
        title: 'Improved Allocation Accuracy',
        description: 'Tag-based allocation has improved accuracy by 5.2% this month',
        impact: 'Better cost visibility and accountability',
        value: 5.2
      },
      {
        id: 'insight-3',
        type: 'info',
        title: 'Unallocated Costs Detected',
        description: '$58,000 in costs could not be allocated due to missing tags',
        impact: 'Reduced cost transparency',
        action: 'Implement comprehensive tagging strategy',
        value: 58000
      },
      {
        id: 'insight-4',
        type: 'success',
        title: 'Cost Savings Identified',
        description: 'Idle resources detected that could save $12,000/month',
        impact: 'Potential annual savings of $144,000',
        action: 'Review and terminate idle resources',
        value: 144000
      }
    ];

    setRules(mockRules);
    setCostCenters(mockCenters);
    setHistory(mockHistory);
    setInsights(mockInsights);
  }, []);

  const handleRunAllocation = useCallback(() => {
    setAllocationRunning(true);
    setTimeout(() => {
      setAllocationRunning(false);
      // Update history with new run
    }, 3000);
  }, []);

  const handleExportData = useCallback(() => {
    console.log('Exporting allocation data...');
  }, []);

  const handleDeleteRule = useCallback((ruleId: string) => {
    setRules(prev => prev.filter(r => r.id !== ruleId));
  }, []);

  const handleToggleRule = useCallback((ruleId: string) => {
    setRules(prev => prev.map(r => 
      r.id === ruleId 
        ? { ...r, status: r.status === 'active' ? 'inactive' : 'active' }
        : r
    ));
  }, []);

  const filteredRules = rules.filter(rule => {
    if (filterStatus !== 'all' && rule.status !== filterStatus) return false;
    if (searchTerm && !rule.name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  const totalUnallocated = history[0]?.unallocatedCost || 0;
  const allocationAccuracy = history[0]?.accuracy || 0;
  const totalAllocated = costCenters.reduce((sum, cc) => sum + cc.allocated, 0);

  // Chart data
  const allocationTrendData = {
    labels: history.slice(0, 6).reverse().map(h => h.period),
    datasets: [
      {
        label: 'Allocated',
        data: history.slice(0, 6).reverse().map(h => h.allocatedCost),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Unallocated',
        data: history.slice(0, 6).reverse().map(h => h.unallocatedCost),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const costCenterDistribution = {
    labels: costCenters.map(cc => cc.name),
    datasets: [{
      data: costCenters.map(cc => cc.allocated),
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(34, 197, 94, 0.8)',
        'rgba(251, 146, 60, 0.8)',
        'rgba(147, 51, 234, 0.8)',
        'rgba(239, 68, 68, 0.8)',
        'rgba(236, 72, 153, 0.8)'
      ],
      borderWidth: 0
    }]
  };

  const rulePerformance = {
    labels: rules.map(r => r.name.split(' ')[0]),
    datasets: [{
      label: 'Accuracy %',
      data: rules.map(r => r.accuracy),
      backgroundColor: rules.map(r => 
        r.status === 'active' ? 'rgba(34, 197, 94, 0.8)' :
        r.status === 'testing' ? 'rgba(251, 146, 60, 0.8)' :
        'rgba(156, 163, 175, 0.8)'
      ),
      borderWidth: 0
    }]
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Share2 className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Cost Allocation Center
              </h1>
              <p className="text-gray-400">Intelligent cost distribution and chargeback management</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Button variant="outline" onClick={handleExportData}>
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </Button>
            <Button 
              onClick={handleRunAllocation} 
              className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              disabled={allocationRunning}
            >
              {allocationRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Calculator className="w-4 h-4 mr-2" />
                  Run Allocation
                </>
              )}
            </Button>
          </div>
        </motion.div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <Card className="bg-gray-900/50 border-gray-800 backdrop-blur">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Costs</p>
                  <p className="text-2xl font-bold">${(totalAllocated / 1000000).toFixed(2)}M</p>
                  <p className="text-xs text-green-400 mt-1">+5.2% from last month</p>
                </div>
                <div className="p-3 bg-blue-500/20 rounded-lg">
                  <DollarSign className="w-6 h-6 text-blue-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800 backdrop-blur">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Allocated</p>
                  <p className="text-2xl font-bold">96.1%</p>
                  <Progress value={96.1} className="w-full h-2 mt-2" />
                </div>
                <div className="p-3 bg-green-500/20 rounded-lg">
                  <PieChart className="w-6 h-6 text-green-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800 backdrop-blur">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Unallocated</p>
                  <p className="text-2xl font-bold text-orange-400">${(totalUnallocated / 1000).toFixed(0)}K</p>
                  <p className="text-xs text-gray-400 mt-1">3.9% of total</p>
                </div>
                <div className="p-3 bg-orange-500/20 rounded-lg">
                  <AlertCircle className="w-6 h-6 text-orange-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800 backdrop-blur">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Accuracy</p>
                  <p className="text-2xl font-bold">{allocationAccuracy.toFixed(1)}%</p>
                  <p className="text-xs text-green-400 mt-1">Above target</p>
                </div>
                <div className="p-3 bg-purple-500/20 rounded-lg">
                  <Target className="w-6 h-6 text-purple-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-900/50 border-gray-800 backdrop-blur">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Active Rules</p>
                  <p className="text-2xl font-bold">{rules.filter(r => r.status === 'active').length}</p>
                  <p className="text-xs text-gray-400 mt-1">of {rules.length} total</p>
                </div>
                <div className="p-3 bg-indigo-500/20 rounded-lg">
                  <Settings className="w-6 h-6 text-indigo-400" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Insights Panel */}
        {insights.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {insights.map((insight) => (
              <Alert 
                key={insight.id}
                className={`border ${
                  insight.type === 'warning' ? 'bg-yellow-900/20 border-yellow-800' :
                  insight.type === 'success' ? 'bg-green-900/20 border-green-800' :
                  insight.type === 'error' ? 'bg-red-900/20 border-red-800' :
                  'bg-blue-900/20 border-blue-800'
                }`}
              >
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  <div className="font-semibold mb-1">{insight.title}</div>
                  <div className="text-sm text-gray-400">{insight.description}</div>
                  {insight.value && (
                    <div className="text-lg font-bold mt-2">
                      {insight.type === 'success' && insight.value < 100 
                        ? `+${insight.value}%`
                        : `$${(insight.value / 1000).toFixed(0)}K`
                      }
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            ))}
          </div>
        )}

        {/* Main Content */}
        <Tabs defaultValue="rules" className="space-y-4">
          <TabsList className="bg-gray-900/50 border-gray-800">
            <TabsTrigger value="rules">Allocation Rules</TabsTrigger>
            <TabsTrigger value="centers">Cost Centers</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="rules" className="space-y-4">
            {/* Rules Toolbar */}
            <Card className="bg-gray-900/50 border-gray-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <Input
                      placeholder="Search rules..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-64 bg-gray-800 border-gray-700"
                    />
                    <Select value={filterStatus} onValueChange={setFilterStatus}>
                      <SelectTrigger className="w-40 bg-gray-800 border-gray-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="active">Active</SelectItem>
                        <SelectItem value="inactive">Inactive</SelectItem>
                        <SelectItem value="draft">Draft</SelectItem>
                        <SelectItem value="testing">Testing</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
                      >
                        {viewMode === 'grid' ? <Grid3x3 className="w-4 h-4" /> : <Layers className="w-4 h-4" />}
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Upload className="w-4 h-4 mr-2" />
                      Import
                    </Button>
                    <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                      <Plus className="w-4 h-4 mr-2" />
                      Add Rule
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Rules Grid */}
            <div className={viewMode === 'grid' ? 'grid grid-cols-1 lg:grid-cols-2 gap-4' : 'space-y-4'}>
              {filteredRules.map((rule) => (
                <motion.div
                  key={rule.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.2 }}
                >
                  <Card className="bg-gray-900/50 border-gray-800 hover:border-gray-700 transition-all">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <CardTitle className="text-lg">{rule.name}</CardTitle>
                          <Badge 
                            variant={rule.status === 'active' ? 'default' : 
                                    rule.status === 'testing' ? 'secondary' : 
                                    rule.status === 'draft' ? 'outline' : 'destructive'}
                          >
                            {rule.status}
                          </Badge>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={() => handleToggleRule(rule.id)}
                          >
                            {rule.status === 'active' ? 
                              <Pause className="w-4 h-4" /> : 
                              <Play className="w-4 h-4" />
                            }
                          </Button>
                          <Button size="icon" variant="ghost">
                            <Edit className="w-4 h-4" />
                          </Button>
                          <Button 
                            size="icon" 
                            variant="ghost"
                            onClick={() => handleDeleteRule(rule.id)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-gray-400">Type</p>
                          <p className="font-medium capitalize flex items-center">
                            {rule.type === 'custom' && <Zap className="w-4 h-4 mr-1 text-purple-400" />}
                            {rule.type === 'tag-based' && <Tags className="w-4 h-4 mr-1 text-blue-400" />}
                            {rule.type}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">Accuracy</p>
                          <div className="flex items-center space-x-2">
                            <p className="font-medium">{rule.accuracy}%</p>
                            <Progress value={rule.accuracy} className="w-16 h-2" />
                          </div>
                        </div>
                        <div>
                          <p className="text-gray-400">Total Allocated</p>
                          <p className="font-medium">${(rule.totalAllocated / 1000000).toFixed(2)}M</p>
                        </div>
                        <div>
                          <p className="text-gray-400">Runs</p>
                          <p className="font-medium">{rule.runsCount}</p>
                        </div>
                      </div>
                      
                      <div>
                        <p className="text-gray-400 text-sm mb-1">Formula</p>
                        <code className="text-xs bg-gray-800 p-2 rounded block text-blue-400">
                          {rule.formula}
                        </code>
                      </div>

                      <div>
                        <p className="text-gray-400 text-sm mb-1">Targets</p>
                        <div className="flex flex-wrap gap-1">
                          {rule.targets.slice(0, 3).map((target, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {target}
                            </Badge>
                          ))}
                          {rule.targets.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{rule.targets.length - 3} more
                            </Badge>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-sm pt-2 border-t border-gray-800">
                        <div className="flex items-center space-x-2 text-gray-400">
                          <Clock className="w-4 h-4" />
                          <span>Next: {new Date(rule.nextRun).toLocaleTimeString()}</span>
                        </div>
                        <Button size="sm" variant="ghost">
                          Test Run
                          <ChevronRight className="w-4 h-4 ml-1" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="centers" className="space-y-4">
            {/* Cost Centers Controls */}
            <Card className="bg-gray-900/50 border-gray-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <Select defaultValue="all">
                      <SelectTrigger className="w-40 bg-gray-800 border-gray-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Departments</SelectItem>
                        <SelectItem value="tech">Technology</SelectItem>
                        <SelectItem value="marketing">Marketing</SelectItem>
                        <SelectItem value="sales">Sales</SelectItem>
                        <SelectItem value="ops">Operations</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setSelectedCenters([])}
                    >
                      Clear Selection
                    </Button>
                  </div>
                  <div className="text-sm text-gray-400">
                    {selectedCenters.length} selected
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Cost Centers Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {costCenters.map((center) => (
                <Card 
                  key={center.id} 
                  className={`bg-gray-900/50 border-gray-800 hover:border-gray-700 transition-all cursor-pointer ${
                    selectedCenters.includes(center.id) ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => {
                    setSelectedCenters(prev => 
                      prev.includes(center.id) 
                        ? prev.filter(id => id !== center.id)
                        : [...prev, center.id]
                    );
                  }}
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-lg flex items-center">
                          {center.name}
                          <Badge variant="outline" className="ml-2 text-xs">
                            {center.code}
                          </Badge>
                        </CardTitle>
                        <p className="text-sm text-gray-400">{center.department}</p>
                      </div>
                      <div className={`p-2 rounded-lg ${
                        center.trend === 'up' ? 'bg-red-500/20' : 
                        center.trend === 'down' ? 'bg-green-500/20' : 
                        'bg-gray-500/20'
                      }`}>
                        {center.trend === 'up' ? 
                          <ArrowUpRight className="w-5 h-5 text-red-400" /> : 
                         center.trend === 'down' ? 
                          <ArrowDownRight className="w-5 h-5 text-green-400" /> : 
                          <Activity className="w-5 h-5 text-gray-400" />
                        }
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-400">Budget vs Allocated</span>
                          <span className="font-medium">
                            ${(center.allocated / 1000).toFixed(0)}K / ${(center.budget / 1000).toFixed(0)}K
                          </span>
                        </div>
                        <Progress 
                          value={(center.allocated / center.budget) * 100} 
                          className="h-2"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-gray-400">Variance</p>
                          <p className={`font-bold text-lg ${
                            center.variance < 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {center.variance > 0 ? '+' : ''}{center.variance}%
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">YTD Total</p>
                          <p className="font-bold text-lg">
                            ${(center.ytd / 1000000).toFixed(1)}M
                          </p>
                        </div>
                      </div>

                      <div>
                        <p className="text-gray-400 text-sm mb-2">Manager</p>
                        <div className="flex items-center space-x-2">
                          <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full" />
                          <span className="text-sm">{center.manager}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-1 pt-3 border-t border-gray-800">
                      {center.tags.map((tag, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Cost Distribution Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle>Cost Distribution by Center</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <Doughnut data={costCenterDistribution} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'right',
                          labels: { color: 'white', font: { size: 11 } }
                        }
                      }
                    }} />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle>Budget vs Actual Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <Bar data={{
                      labels: costCenters.map(cc => cc.name),
                      datasets: [
                        {
                          label: 'Budget',
                          data: costCenters.map(cc => cc.budget),
                          backgroundColor: 'rgba(59, 130, 246, 0.5)'
                        },
                        {
                          label: 'Allocated',
                          data: costCenters.map(cc => cc.allocated),
                          backgroundColor: 'rgba(147, 51, 234, 0.8)'
                        }
                      ]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          ticks: { color: 'white' },
                          grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                          ticks: { color: 'white' },
                          grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                      },
                      plugins: {
                        legend: { labels: { color: 'white' } }
                      }
                    }} />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="history" className="space-y-4">
            {/* Allocation History Table */}
            <Card className="bg-gray-900/50 border-gray-800">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Allocation Run History</CardTitle>
                  <Select defaultValue="30d">
                    <SelectTrigger className="w-32 bg-gray-800 border-gray-700">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="7d">Last 7 days</SelectItem>
                      <SelectItem value="30d">Last 30 days</SelectItem>
                      <SelectItem value="90d">Last 90 days</SelectItem>
                      <SelectItem value="1y">Last year</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-800">
                        <th className="text-left p-3 text-gray-400 font-medium">Period</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Total Cost</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Allocated</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Unallocated</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Accuracy</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Duration</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Status</th>
                        <th className="text-left p-3 text-gray-400 font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((run) => (
                        <tr key={run.id} className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                          <td className="p-3 font-medium">{run.period}</td>
                          <td className="p-3">${(run.totalCost / 1000000).toFixed(2)}M</td>
                          <td className="p-3 text-green-400">${(run.allocatedCost / 1000000).toFixed(2)}M</td>
                          <td className="p-3 text-orange-400">${(run.unallocatedCost / 1000).toFixed(0)}K</td>
                          <td className="p-3">
                            <div className="flex items-center space-x-2">
                              <span>{run.accuracy.toFixed(1)}%</span>
                              <Progress value={run.accuracy} className="w-16 h-2" />
                            </div>
                          </td>
                          <td className="p-3 text-gray-400">{run.duration}s</td>
                          <td className="p-3">
                            <Badge variant={
                              run.status === 'completed' ? 'default' : 
                              run.status === 'processing' ? 'secondary' : 
                              run.status === 'partial' ? 'outline' : 'destructive'
                            }>
                              {run.status}
                            </Badge>
                          </td>
                          <td className="p-3">
                            <div className="flex items-center space-x-2">
                              <Button size="sm" variant="ghost">View</Button>
                              <Button size="sm" variant="ghost">
                                <Copy className="w-4 h-4" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Trend Chart */}
            <Card className="bg-gray-900/50 border-gray-800">
              <CardHeader>
                <CardTitle>Allocation Trend Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <Line data={allocationTrendData} options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                      },
                      x: {
                        ticks: { color: 'white' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                      }
                    },
                    plugins: {
                      legend: {
                        labels: { color: 'white' }
                      }
                    }
                  }} />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Rule Performance */}
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle>Rule Performance Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <Bar data={rulePerformance} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          min: 90,
                          max: 100,
                          ticks: { color: 'white' },
                          grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                          ticks: { color: 'white' },
                          grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                      },
                      plugins: {
                        legend: { display: false }
                      }
                    }} />
                  </div>
                </CardContent>
              </Card>

              {/* Accuracy Over Time */}
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle>Allocation Accuracy Over Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <Line data={{
                      labels: history.slice(0, 6).reverse().map(h => h.period),
                      datasets: [{
                        label: 'Accuracy %',
                        data: history.slice(0, 6).reverse().map(h => h.accuracy),
                        borderColor: 'rgb(147, 51, 234)',
                        backgroundColor: 'rgba(147, 51, 234, 0.1)',
                        fill: true,
                        tension: 0.4
                      }]
                    }} options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          min: 90,
                          max: 100,
                          ticks: { color: 'white' },
                          grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                          ticks: { color: 'white' },
                          grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                      },
                      plugins: {
                        legend: {
                          labels: { color: 'white' }
                        }
                      }
                    }} />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-base">Average Allocation Time</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <p className="text-4xl font-bold text-blue-400">72s</p>
                    <p className="text-sm text-gray-400 mt-2">-15% from last month</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-base">Cost Recovery Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <p className="text-4xl font-bold text-green-400">96.1%</p>
                    <p className="text-sm text-gray-400 mt-2">+2.3% improvement</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-base">Tag Coverage</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <p className="text-4xl font-bold text-purple-400">89%</p>
                    <p className="text-sm text-gray-400 mt-2">11% resources untagged</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="settings" className="space-y-4">
            <Card className="bg-gray-900/50 border-gray-800">
              <CardHeader>
                <CardTitle>Allocation Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Automatic Allocation</Label>
                        <p className="text-sm text-gray-400">Run allocation automatically on schedule</p>
                      </div>
                      <Switch checked={autoAllocate} onCheckedChange={setAutoAllocate} />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Preview Mode</Label>
                        <p className="text-sm text-gray-400">Test allocations without applying changes</p>
                      </div>
                      <Switch checked={showPreview} onCheckedChange={setShowPreview} />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Email Notifications</Label>
                        <p className="text-sm text-gray-400">Send reports after each allocation run</p>
                      </div>
                      <Switch defaultChecked />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Anomaly Detection</Label>
                        <p className="text-sm text-gray-400">Alert on unusual allocation patterns</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Allocation Schedule</Label>
                      <Select defaultValue="daily">
                        <SelectTrigger className="bg-gray-800 border-gray-700">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="hourly">Hourly</SelectItem>
                          <SelectItem value="daily">Daily at midnight</SelectItem>
                          <SelectItem value="weekly">Weekly on Sunday</SelectItem>
                          <SelectItem value="monthly">Monthly on 1st</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Unallocated Cost Threshold</Label>
                      <div className="flex items-center space-x-2">
                        <Input type="number" defaultValue="5" className="w-20 bg-gray-800 border-gray-700" />
                        <span className="text-gray-400">%</span>
                      </div>
                      <p className="text-sm text-gray-400">Alert when unallocated costs exceed this threshold</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Default Allocation Method</Label>
                      <Select defaultValue="usage">
                        <SelectTrigger className="bg-gray-800 border-gray-700">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="usage">Usage-based</SelectItem>
                          <SelectItem value="percentage">Percentage</SelectItem>
                          <SelectItem value="fixed">Fixed Amount</SelectItem>
                          <SelectItem value="tag-based">Tag-based</SelectItem>
                          <SelectItem value="custom">Custom Formula</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Retention Period</Label>
                      <Select defaultValue="90">
                        <SelectTrigger className="bg-gray-800 border-gray-700">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="30">30 days</SelectItem>
                          <SelectItem value="90">90 days</SelectItem>
                          <SelectItem value="180">180 days</SelectItem>
                          <SelectItem value="365">1 year</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end space-x-2 pt-4 border-t border-gray-800">
                  <Button variant="outline">Reset Defaults</Button>
                  <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                    Save Settings
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}