'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  TrendingUp,
  TrendingDown,
  Shield,
  AlertTriangle,
  CheckCircle,
  DollarSign,
  Activity,
  Users,
  Cloud,
  Target,
  BarChart3,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  Zap,
  Award,
  Globe,
  Lock,
  Gauge,
  Info
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RePieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar
} from 'recharts';

interface KPI {
  id: string;
  name: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  target: number;
  status: 'on-track' | 'at-risk' | 'off-track';
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

interface ComplianceMetric {
  category: string;
  score: number;
  change: number;
  issues: number;
}

interface CostMetric {
  month: string;
  actual: number;
  budget: number;
  forecast: number;
  savings: number;
}

interface RiskItem {
  id: string;
  title: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  impact: string;
  likelihood: string;
  mitigation: string;
}

export default function ExecutiveDashboard() {
  const [loading, setLoading] = useState(true);
  const [kpis, setKpis] = useState<KPI[]>([]);
  const [complianceData, setComplianceData] = useState<ComplianceMetric[]>([]);
  const [costTrends, setCostTrends] = useState<CostMetric[]>([]);
  const [risks, setRisks] = useState<RiskItem[]>([]);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [resourceDistribution, setResourceDistribution] = useState<any[]>([]);

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock KPIs
      setKpis([
        {
          id: 'compliance-rate',
          name: 'Overall Compliance',
          value: '94.2%',
          change: 3.5,
          trend: 'up',
          target: 95,
          status: 'at-risk',
          icon: Shield,
          color: '#3B82F6'
        },
        {
          id: 'cost-savings',
          name: 'Cost Savings YTD',
          value: '$2.4M',
          change: 18.2,
          trend: 'up',
          target: 100,
          status: 'on-track',
          icon: DollarSign,
          color: '#10B981'
        },
        {
          id: 'incidents-prevented',
          name: 'Incidents Prevented',
          value: 1842,
          change: -12.5,
          trend: 'down',
          target: 90,
          status: 'on-track',
          icon: AlertTriangle,
          color: '#F59E0B'
        },
        {
          id: 'resource-efficiency',
          name: 'Resource Efficiency',
          value: '87%',
          change: 5.8,
          trend: 'up',
          target: 85,
          status: 'on-track',
          icon: Activity,
          color: '#8B5CF6'
        },
        {
          id: 'security-score',
          name: 'Security Score',
          value: '92/100',
          change: 2.1,
          trend: 'up',
          target: 90,
          status: 'on-track',
          icon: Lock,
          color: '#EF4444'
        },
        {
          id: 'automation-rate',
          name: 'Automation Rate',
          value: '76%',
          change: 8.3,
          trend: 'up',
          target: 80,
          status: 'at-risk',
          icon: Zap,
          color: '#06B6D4'
        }
      ]);

      // Mock compliance data
      setComplianceData([
        { category: 'SOC2', score: 96, change: 2, issues: 3 },
        { category: 'ISO 27001', score: 94, change: 5, issues: 7 },
        { category: 'HIPAA', score: 91, change: -1, issues: 12 },
        { category: 'PCI DSS', score: 98, change: 3, issues: 2 },
        { category: 'GDPR', score: 93, change: 4, issues: 8 }
      ]);

      // Mock cost trends
      setCostTrends([
        { month: 'Jan', actual: 320000, budget: 350000, forecast: 330000, savings: 30000 },
        { month: 'Feb', actual: 310000, budget: 350000, forecast: 320000, savings: 40000 },
        { month: 'Mar', actual: 340000, budget: 350000, forecast: 345000, savings: 10000 },
        { month: 'Apr', actual: 335000, budget: 360000, forecast: 340000, savings: 25000 },
        { month: 'May', actual: 325000, budget: 360000, forecast: 330000, savings: 35000 },
        { month: 'Jun', actual: 345000, budget: 370000, forecast: 350000, savings: 25000 },
        { month: 'Jul', actual: 355000, budget: 370000, forecast: 360000, savings: 15000 },
        { month: 'Aug', actual: 360000, budget: 380000, forecast: 365000, savings: 20000 },
        { month: 'Sep', actual: 365000, budget: 380000, forecast: 370000, savings: 15000 },
        { month: 'Oct', actual: 370000, budget: 390000, forecast: 375000, savings: 20000 }
      ]);

      // Mock risks
      setRisks([
        {
          id: '1',
          title: 'Data breach vulnerability in legacy systems',
          category: 'Security',
          severity: 'critical',
          impact: 'High',
          likelihood: 'Medium',
          mitigation: 'Immediate patching and system upgrade required'
        },
        {
          id: '2',
          title: 'Non-compliance with new GDPR requirements',
          category: 'Compliance',
          severity: 'high',
          impact: 'High',
          likelihood: 'Low',
          mitigation: 'Update data processing procedures by Q2'
        },
        {
          id: '3',
          title: 'Cloud cost overrun projection',
          category: 'Financial',
          severity: 'medium',
          impact: 'Medium',
          likelihood: 'High',
          mitigation: 'Implement cost optimization strategies'
        },
        {
          id: '4',
          title: 'Key personnel dependency',
          category: 'Operational',
          severity: 'medium',
          impact: 'Medium',
          likelihood: 'Medium',
          mitigation: 'Cross-training and documentation initiative'
        }
      ]);

      // Mock performance data
      setPerformanceData([
        { metric: 'Availability', value: 99.95, target: 99.9 },
        { metric: 'Response Time', value: 125, target: 150 },
        { metric: 'Error Rate', value: 0.12, target: 0.5 },
        { metric: 'Throughput', value: 8500, target: 8000 }
      ]);

      // Mock resource distribution
      setResourceDistribution([
        { name: 'Production', value: 45, color: '#3B82F6' },
        { name: 'Development', value: 20, color: '#10B981' },
        { name: 'Staging', value: 15, color: '#F59E0B' },
        { name: 'Testing', value: 12, color: '#8B5CF6' },
        { name: 'Backup', value: 8, color: '#EF4444' }
      ]);

      setLoading(false);
    }, 1000);
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(value);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'high':
        return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';
      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'on-track':
        return 'text-green-600 dark:text-green-400';
      case 'at-risk':
        return 'text-amber-600 dark:text-amber-400';
      case 'off-track':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading executive dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <Gauge className="h-8 w-8 text-blue-600" />
              Executive Dashboard
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Real-time KPIs and strategic insights for cloud governance
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-sm text-gray-500 dark:text-gray-400">Last Updated</div>
              <div className="font-medium">{new Date().toLocaleString()}</div>
            </div>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              Export Report
            </button>
          </div>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {kpis.map((kpi) => {
          const Icon = kpi.icon;
          return (
            <Card key={kpi.id} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <span style={{ color: kpi.color }}>
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                    kpi.status === 'on-track' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                    kpi.status === 'at-risk' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400' :
                    'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                  }`}>
                    {kpi.status.replace('-', ' ')}
                  </span>
                </div>
                <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400 mt-2">
                  {kpi.name}
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-2">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {kpi.value}
                </div>
                <div className={`flex items-center gap-1 text-sm mt-1 ${
                  kpi.trend === 'up' ? 'text-green-600 dark:text-green-400' :
                  kpi.trend === 'down' ? 'text-red-600 dark:text-red-400' :
                  'text-gray-600 dark:text-gray-400'
                }`}>
                  {kpi.trend === 'up' && <ArrowUpRight className="h-3 w-3" />}
                  {kpi.trend === 'down' && <ArrowDownRight className="h-3 w-3" />}
                  {kpi.change > 0 ? '+' : ''}{kpi.change}%
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Trends */}
        <Card>
          <CardHeader>
            <CardTitle>Cost Management Overview</CardTitle>
            <CardDescription>Monthly cloud spend vs budget with savings</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={costTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value: number) => formatCurrency(value)} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="budget"
                  stackId="1"
                  stroke="#94A3B8"
                  fill="#E2E8F0"
                  name="Budget"
                />
                <Area
                  type="monotone"
                  dataKey="actual"
                  stackId="2"
                  stroke="#3B82F6"
                  fill="#93C5FD"
                  name="Actual"
                />
                <Line
                  type="monotone"
                  dataKey="savings"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Savings"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Compliance Scores */}
        <Card>
          <CardHeader>
            <CardTitle>Compliance Framework Scores</CardTitle>
            <CardDescription>Current compliance status across frameworks</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="90%" data={complianceData}>
                <RadialBar
                  background
                  dataKey="score"
                  fill="#3B82F6"
                />
                <Legend />
                <Tooltip />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-5 gap-2 mt-4">
              {complianceData.map((item) => (
                <div key={item.category} className="text-center">
                  <div className="text-xs text-gray-600 dark:text-gray-400">{item.category}</div>
                  <div className="text-lg font-bold">{item.score}%</div>
                  <div className={`text-xs ${
                    item.change > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {item.change > 0 ? '+' : ''}{item.change}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Matrix and Resource Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Matrix */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Risk Assessment Matrix</CardTitle>
            <CardDescription>Top organizational risks requiring attention</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {risks.map((risk) => (
                <div key={risk.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium">{risk.title}</h4>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(risk.severity)}`}>
                          {risk.severity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{risk.category}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Impact:</span>
                      <span className="ml-2 font-medium">{risk.impact}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Likelihood:</span>
                      <span className="ml-2 font-medium">{risk.likelihood}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Mitigation:</span>
                    </div>
                  </div>
                  <div className="mt-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-sm">
                    {risk.mitigation}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Resource Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Resource Distribution</CardTitle>
            <CardDescription>Infrastructure allocation by environment</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <RePieChart>
                <Pie
                  data={resourceDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {resourceDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `${value}%`} />
              </RePieChart>
            </ResponsiveContainer>
            <div className="space-y-2 mt-4">
              {resourceDistribution.map((item) => (
                <div key={item.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-sm">{item.name}</span>
                  </div>
                  <span className="text-sm font-medium">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>System Performance Metrics</CardTitle>
          <CardDescription>Real-time performance against SLA targets</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {performanceData.map((metric) => (
              <div key={metric.metric} className="text-center">
                <div className="relative inline-flex items-center justify-center w-32 h-32">
                  <svg className="w-32 h-32 transform -rotate-90">
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="none"
                      className="text-gray-200 dark:text-gray-700"
                    />
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="none"
                      strokeDasharray={`${(metric.value / metric.target) * 352} 352`}
                      className="text-blue-600"
                    />
                  </svg>
                  <div className="absolute">
                    <div className="text-2xl font-bold">{metric.value}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      / {metric.target}
                    </div>
                  </div>
                </div>
                <div className="mt-2 font-medium">{metric.metric}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-900 border-blue-200 dark:border-blue-800">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-600" />
            Strategic Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Target className="h-6 w-6 text-blue-600 mb-2" />
              <div className="font-medium">Set OKRs</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Define quarterly objectives
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Award className="h-6 w-6 text-green-600 mb-2" />
              <div className="font-medium">Review Performance</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Monthly KPI review
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Globe className="h-6 w-6 text-purple-600 mb-2" />
              <div className="font-medium">Compliance Audit</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Schedule audit review
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Shield className="h-6 w-6 text-red-600 mb-2" />
              <div className="font-medium">Risk Assessment</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Quarterly risk review
              </div>
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}