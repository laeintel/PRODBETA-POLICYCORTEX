'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  TrendingUp,
  DollarSign,
  PieChart,
  BarChart3,
  Calculator,
  Target,
  Zap,
  Shield,
  Clock,
  Users,
  ArrowUpRight,
  ArrowDownRight,
  AlertCircle,
  CheckCircle2,
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
  ResponsiveContainer
} from 'recharts';

interface ROIMetric {
  id: string;
  name: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'neutral';
  unit: 'currency' | 'percentage' | 'number' | 'days';
}

interface CostBreakdown {
  category: string;
  actual: number;
  optimized: number;
  savings: number;
  percentage: number;
}

interface ROISummary {
  totalSavings: number;
  savingsPercentage: number;
  paybackPeriod: number;
  netPresentValue: number;
  internalRateOfReturn: number;
  breakEvenPoint: Date;
}

export default function ROICalculator() {
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState<'monthly' | 'quarterly' | 'yearly'>('quarterly');
  const [metrics, setMetrics] = useState<ROIMetric[]>([]);
  const [breakdown, setBreakdown] = useState<CostBreakdown[]>([]);
  const [summary, setSummary] = useState<ROISummary | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  const [savingsDistribution, setSavingsDistribution] = useState<any[]>([]);

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      // Mock ROI metrics
      setMetrics([
        {
          id: 'total-savings',
          name: 'Total Cost Savings',
          value: 1245000,
          change: 23.5,
          trend: 'up',
          unit: 'currency'
        },
        {
          id: 'compliance-rate',
          name: 'Compliance Rate',
          value: 96.8,
          change: 4.2,
          trend: 'up',
          unit: 'percentage'
        },
        {
          id: 'mttr',
          name: 'Mean Time to Resolution',
          value: 2.4,
          change: -35,
          trend: 'down',
          unit: 'days'
        },
        {
          id: 'automation-rate',
          name: 'Automation Rate',
          value: 78,
          change: 12,
          trend: 'up',
          unit: 'percentage'
        },
        {
          id: 'prevented-incidents',
          name: 'Prevented Incidents',
          value: 342,
          change: 45,
          trend: 'up',
          unit: 'number'
        },
        {
          id: 'resource-efficiency',
          name: 'Resource Efficiency',
          value: 89,
          change: 8,
          trend: 'up',
          unit: 'percentage'
        }
      ]);

      // Mock cost breakdown
      setBreakdown([
        {
          category: 'Compute Resources',
          actual: 450000,
          optimized: 320000,
          savings: 130000,
          percentage: 28.9
        },
        {
          category: 'Storage & Backup',
          actual: 280000,
          optimized: 195000,
          savings: 85000,
          percentage: 30.4
        },
        {
          category: 'Network & CDN',
          actual: 180000,
          optimized: 135000,
          savings: 45000,
          percentage: 25.0
        },
        {
          category: 'Licenses & Subscriptions',
          actual: 320000,
          optimized: 240000,
          savings: 80000,
          percentage: 25.0
        },
        {
          category: 'Security & Compliance',
          actual: 150000,
          optimized: 120000,
          savings: 30000,
          percentage: 20.0
        },
        {
          category: 'Operations & Support',
          actual: 220000,
          optimized: 165000,
          savings: 55000,
          percentage: 25.0
        }
      ]);

      // Mock ROI summary
      setSummary({
        totalSavings: 1245000,
        savingsPercentage: 26.8,
        paybackPeriod: 8.5,
        netPresentValue: 3450000,
        internalRateOfReturn: 42.5,
        breakEvenPoint: new Date('2024-03-15')
      });

      // Mock chart data for savings over time
      setChartData([
        { month: 'Jan', projected: 80000, actual: 95000, cumulative: 95000 },
        { month: 'Feb', projected: 85000, actual: 102000, cumulative: 197000 },
        { month: 'Mar', projected: 90000, actual: 110000, cumulative: 307000 },
        { month: 'Apr', projected: 95000, actual: 118000, cumulative: 425000 },
        { month: 'May', projected: 100000, actual: 125000, cumulative: 550000 },
        { month: 'Jun', projected: 105000, actual: 132000, cumulative: 682000 },
        { month: 'Jul', projected: 110000, actual: 138000, cumulative: 820000 },
        { month: 'Aug', projected: 115000, actual: 142000, cumulative: 962000 },
        { month: 'Sep', projected: 120000, actual: 148000, cumulative: 1110000 },
        { month: 'Oct', projected: 125000, actual: 135000, cumulative: 1245000 }
      ]);

      // Mock savings distribution
      setSavingsDistribution([
        { name: 'Cloud Optimization', value: 35, color: '#3B82F6' },
        { name: 'Automation', value: 25, color: '#10B981' },
        { name: 'Policy Enforcement', value: 20, color: '#F59E0B' },
        { name: 'Incident Prevention', value: 15, color: '#EF4444' },
        { name: 'Process Improvement', value: 5, color: '#8B5CF6' }
      ]);

      setLoading(false);
    }, 1000);
  }, [timeframe]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(value);
  };

  const formatMetricValue = (value: number, unit: string) => {
    switch (unit) {
      case 'currency':
        return formatCurrency(value);
      case 'percentage':
        return `${value}%`;
      case 'days':
        return `${value} days`;
      default:
        return value.toLocaleString();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading ROI metrics...</p>
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
              <Calculator className="h-8 w-8 text-blue-600" />
              ROI Calculator & Metrics
            </h1>
            <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
              Track cost savings, efficiency gains, and return on investment
            </p>
          </div>
          <div className="flex gap-2">
            {(['monthly', 'quarterly', 'yearly'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setTimeframe(period)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  timeframe === period
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {period.charAt(0).toUpperCase() + period.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ROI Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <Card className="bg-gradient-to-br from-green-500 to-emerald-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-green-100">Total Savings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCurrency(summary.totalSavings)}</div>
              <div className="text-sm text-green-100 mt-1">
                {summary.savingsPercentage}% reduction
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-blue-500 to-cyan-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-blue-100">NPV</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCurrency(summary.netPresentValue)}</div>
              <div className="text-sm text-blue-100 mt-1">Net present value</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500 to-pink-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-purple-100">IRR</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.internalRateOfReturn}%</div>
              <div className="text-sm text-purple-100 mt-1">Internal rate of return</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-amber-500 to-orange-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-amber-100">Payback Period</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.paybackPeriod} months</div>
              <div className="text-sm text-amber-100 mt-1">Time to ROI</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-indigo-500 to-blue-600 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-indigo-100">Break-Even</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {summary.breakEvenPoint.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
              </div>
              <div className="text-sm text-indigo-100 mt-1">Projected date</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric) => (
          <Card key={metric.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{metric.name}</CardTitle>
                {metric.trend === 'up' ? (
                  <ArrowUpRight className="h-5 w-5 text-green-500" />
                ) : metric.trend === 'down' ? (
                  <ArrowDownRight className="h-5 w-5 text-red-500" />
                ) : (
                  <div className="h-5 w-5" />
                )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-gray-900 dark:text-white">
                {formatMetricValue(metric.value, metric.unit)}
              </div>
              <div className={`text-sm mt-2 ${
                metric.trend === 'up' ? 'text-green-600 dark:text-green-400' : 
                metric.trend === 'down' ? 'text-red-600 dark:text-red-400' :
                'text-gray-600 dark:text-gray-400'
              }`}>
                {metric.trend === 'up' ? '+' : ''}{metric.change}% from last period
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Savings Over Time */}
        <Card>
          <CardHeader>
            <CardTitle>Savings Over Time</CardTitle>
            <CardDescription>Monthly projected vs actual savings with cumulative total</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value: number) => formatCurrency(value)} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="projected"
                  stackId="1"
                  stroke="#94A3B8"
                  fill="#E2E8F0"
                  name="Projected"
                />
                <Area
                  type="monotone"
                  dataKey="actual"
                  stackId="2"
                  stroke="#10B981"
                  fill="#86EFAC"
                  name="Actual"
                />
                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  name="Cumulative"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Savings Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Savings Distribution</CardTitle>
            <CardDescription>Breakdown of savings by optimization category</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RePieChart>
                <Pie
                  data={savingsDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {savingsDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `${value}%`} />
                <Legend />
              </RePieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Cost Breakdown Table */}
      <Card>
        <CardHeader>
          <CardTitle>Cost Optimization Breakdown</CardTitle>
          <CardDescription>Detailed analysis by category with optimization potential</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4">Category</th>
                  <th className="text-right py-3 px-4">Current Cost</th>
                  <th className="text-right py-3 px-4">Optimized Cost</th>
                  <th className="text-right py-3 px-4">Savings</th>
                  <th className="text-right py-3 px-4">Reduction %</th>
                  <th className="text-center py-3 px-4">Status</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.map((item) => (
                  <tr key={item.category} className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4 font-medium">{item.category}</td>
                    <td className="text-right py-3 px-4">{formatCurrency(item.actual)}</td>
                    <td className="text-right py-3 px-4">{formatCurrency(item.optimized)}</td>
                    <td className="text-right py-3 px-4 text-green-600 dark:text-green-400 font-medium">
                      {formatCurrency(item.savings)}
                    </td>
                    <td className="text-right py-3 px-4">
                      <span className="inline-flex items-center gap-1">
                        {item.percentage}%
                        <ArrowDownRight className="h-4 w-4 text-green-500" />
                      </span>
                    </td>
                    <td className="text-center py-3 px-4">
                      {item.percentage >= 30 ? (
                        <span className="inline-flex items-center gap-1 text-green-600 dark:text-green-400">
                          <CheckCircle2 className="h-4 w-4" />
                          Optimized
                        </span>
                      ) : item.percentage >= 20 ? (
                        <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400">
                          <AlertCircle className="h-4 w-4" />
                          In Progress
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-gray-600 dark:text-gray-400">
                          <Info className="h-4 w-4" />
                          Planned
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
                <tr className="font-bold">
                  <td className="py-3 px-4">Total</td>
                  <td className="text-right py-3 px-4">
                    {formatCurrency(breakdown.reduce((sum, item) => sum + item.actual, 0))}
                  </td>
                  <td className="text-right py-3 px-4">
                    {formatCurrency(breakdown.reduce((sum, item) => sum + item.optimized, 0))}
                  </td>
                  <td className="text-right py-3 px-4 text-green-600 dark:text-green-400">
                    {formatCurrency(breakdown.reduce((sum, item) => sum + item.savings, 0))}
                  </td>
                  <td className="text-right py-3 px-4">
                    {(
                      (breakdown.reduce((sum, item) => sum + item.savings, 0) /
                        breakdown.reduce((sum, item) => sum + item.actual, 0)) *
                      100
                    ).toFixed(1)}%
                  </td>
                  <td></td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-900 border-blue-200 dark:border-blue-800">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-600" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Target className="h-6 w-6 text-blue-600 mb-2" />
              <div className="font-medium">Set Cost Targets</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Define budget goals and thresholds
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <PieChart className="h-6 w-6 text-green-600 mb-2" />
              <div className="font-medium">Generate Report</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Export detailed ROI analysis
              </div>
            </button>
            <button className="p-4 bg-white dark:bg-gray-800 rounded-lg hover:shadow-md transition-shadow text-left">
              <Shield className="h-6 w-6 text-purple-600 mb-2" />
              <div className="font-medium">Schedule Review</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Book ROI assessment meeting
              </div>
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}