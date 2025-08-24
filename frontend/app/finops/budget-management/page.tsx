'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/dialog';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadialBarChart, RadialBar, PolarAngleAxis
} from 'recharts';
import {
  DollarSign, TrendingUp, AlertTriangle, Bell, Plus,
  Edit, Trash2, Target, Calendar, Users, Layers,
  AlertCircle, CheckCircle, XCircle, Clock, Settings,
  ArrowUpRight, ArrowDownRight, BellRing, Mail, Slack
} from 'lucide-react';

interface Budget {
  id: string;
  name: string;
  type: 'annual' | 'quarterly' | 'monthly' | 'project';
  amount: number;
  spent: number;
  remaining: number;
  percentage: number;
  status: 'on-track' | 'at-risk' | 'exceeded';
  period: string;
  owner: string;
  department: string;
  alerts: BudgetAlert[];
  forecast: number;
  lastUpdated: string;
}

interface BudgetAlert {
  id: string;
  type: 'warning' | 'critical' | 'info';
  threshold: number;
  triggered: boolean;
  message: string;
  channels: string[];
}

interface BudgetRule {
  id: string;
  name: string;
  condition: string;
  action: string;
  enabled: boolean;
  lastTriggered: string | null;
  triggerCount: number;
}

interface SpendingTrend {
  date: string;
  actual: number;
  budget: number;
  forecast: number;
}

const BudgetManagementPage = () => {
  const [budgets, setBudgets] = useState<Budget[]>([]);
  const [selectedBudget, setSelectedBudget] = useState<Budget | null>(null);
  const [rules, setRules] = useState<BudgetRule[]>([]);
  const [showCreateBudget, setShowCreateBudget] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  // New budget form state
  const [newBudget, setNewBudget] = useState({
    name: '',
    type: 'monthly',
    amount: '',
    department: '',
    owner: '',
    alertThresholds: {
      warning: 80,
      critical: 95
    }
  });

  useEffect(() => {
    // Generate mock data
    const mockBudgets: Budget[] = [
      {
        id: 'BUD-001',
        name: 'Q1 2025 Cloud Infrastructure',
        type: 'quarterly',
        amount: 1500000,
        spent: 1234567,
        remaining: 265433,
        percentage: 82.3,
        status: 'at-risk',
        period: 'Q1 2025',
        owner: 'John Smith',
        department: 'Engineering',
        alerts: [
          {
            id: 'ALT-001',
            type: 'warning',
            threshold: 80,
            triggered: true,
            message: 'Budget utilization exceeded 80%',
            channels: ['email', 'slack']
          }
        ],
        forecast: 1480000,
        lastUpdated: '2 hours ago'
      },
      {
        id: 'BUD-002',
        name: 'Marketing Campaign AWS',
        type: 'project',
        amount: 250000,
        spent: 198450,
        remaining: 51550,
        percentage: 79.4,
        status: 'on-track',
        period: 'Jan - Mar 2025',
        owner: 'Sarah Johnson',
        department: 'Marketing',
        alerts: [],
        forecast: 245000,
        lastUpdated: '1 hour ago'
      },
      {
        id: 'BUD-003',
        name: 'Monthly DevOps Budget',
        type: 'monthly',
        amount: 85000,
        spent: 92340,
        remaining: -7340,
        percentage: 108.6,
        status: 'exceeded',
        period: 'January 2025',
        owner: 'Mike Chen',
        department: 'DevOps',
        alerts: [
          {
            id: 'ALT-002',
            type: 'critical',
            threshold: 100,
            triggered: true,
            message: 'Budget exceeded by $7,340',
            channels: ['email', 'slack', 'teams']
          }
        ],
        forecast: 93000,
        lastUpdated: '30 minutes ago'
      },
      {
        id: 'BUD-004',
        name: 'Data Analytics Platform',
        type: 'annual',
        amount: 3200000,
        spent: 876543,
        remaining: 2323457,
        percentage: 27.4,
        status: 'on-track',
        period: '2025',
        owner: 'Lisa Wang',
        department: 'Data Science',
        alerts: [],
        forecast: 3150000,
        lastUpdated: '4 hours ago'
      },
      {
        id: 'BUD-005',
        name: 'Security & Compliance',
        type: 'quarterly',
        amount: 450000,
        spent: 423890,
        remaining: 26110,
        percentage: 94.2,
        status: 'at-risk',
        period: 'Q1 2025',
        owner: 'David Brown',
        department: 'Security',
        alerts: [
          {
            id: 'ALT-003',
            type: 'critical',
            threshold: 95,
            triggered: false,
            message: 'Approaching budget limit',
            channels: ['email', 'pagerduty']
          }
        ],
        forecast: 448000,
        lastUpdated: '1 hour ago'
      }
    ];

    const mockRules: BudgetRule[] = [
      {
        id: 'RULE-001',
        name: 'Auto-scale Down Non-Prod',
        condition: 'Budget utilization > 90%',
        action: 'Scale down non-production environments',
        enabled: true,
        lastTriggered: '2 days ago',
        triggerCount: 3
      },
      {
        id: 'RULE-002',
        name: 'Notify Executives',
        condition: 'Monthly spend > $100,000',
        action: 'Send executive summary email',
        enabled: true,
        lastTriggered: '1 week ago',
        triggerCount: 12
      },
      {
        id: 'RULE-003',
        name: 'Block New Resources',
        condition: 'Budget exceeded',
        action: 'Block creation of new cloud resources',
        enabled: false,
        lastTriggered: null,
        triggerCount: 0
      },
      {
        id: 'RULE-004',
        name: 'Weekend Shutdown',
        condition: 'Weekend AND environment = dev',
        action: 'Shutdown development resources',
        enabled: true,
        lastTriggered: '3 days ago',
        triggerCount: 8
      }
    ];

    setBudgets(mockBudgets);
    setRules(mockRules);
    setSelectedBudget(mockBudgets[0]);
    setLoading(false);
  }, []);

  // Generate spending trend data
  const generateSpendingTrend = (): SpendingTrend[] => {
    const data: SpendingTrend[] = [];
    const days = 30;
    const baseSpend = 40000;
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        actual: baseSpend + Math.random() * 10000 - 5000 + (i * 500),
        budget: 50000,
        forecast: baseSpend + (i * 600) + Math.random() * 5000
      });
    }
    
    return data;
  };

  const spendingTrend = generateSpendingTrend();

  // Department spending data
  const departmentSpending = [
    { department: 'Engineering', budget: 500000, spent: 423450, percentage: 84.7 },
    { department: 'Marketing', budget: 250000, spent: 198450, percentage: 79.4 },
    { department: 'Data Science', budget: 350000, spent: 287340, percentage: 82.1 },
    { department: 'DevOps', budget: 200000, spent: 192340, percentage: 96.2 },
    { department: 'Security', budget: 150000, spent: 141230, percentage: 94.2 },
    { department: 'Sales', budget: 100000, spent: 67890, percentage: 67.9 }
  ];

  // Budget utilization by type
  const utilizationByType = [
    { name: 'Annual', value: 72, fill: '#8884d8' },
    { name: 'Quarterly', value: 88, fill: '#83a6ed' },
    { name: 'Monthly', value: 95, fill: '#8dd1e1' },
    { name: 'Project', value: 79, fill: '#82ca9d' }
  ];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'on-track':
        return 'text-green-600 bg-green-100 dark:bg-green-900/50';
      case 'at-risk':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/50';
      case 'exceeded':
        return 'text-red-600 bg-red-100 dark:bg-red-900/50';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-900/50';
    }
  };

  const handleCreateBudget = () => {
    const budget: Budget = {
      id: `BUD-${Date.now()}`,
      name: newBudget.name,
      type: newBudget.type as any,
      amount: parseFloat(newBudget.amount),
      spent: 0,
      remaining: parseFloat(newBudget.amount),
      percentage: 0,
      status: 'on-track',
      period: 'Current',
      owner: newBudget.owner,
      department: newBudget.department,
      alerts: [
        {
          id: `ALT-${Date.now()}-1`,
          type: 'warning',
          threshold: newBudget.alertThresholds.warning,
          triggered: false,
          message: `Budget utilization exceeded ${newBudget.alertThresholds.warning}%`,
          channels: ['email', 'slack']
        },
        {
          id: `ALT-${Date.now()}-2`,
          type: 'critical',
          threshold: newBudget.alertThresholds.critical,
          triggered: false,
          message: `Budget utilization exceeded ${newBudget.alertThresholds.critical}%`,
          channels: ['email', 'slack', 'pagerduty']
        }
      ],
      forecast: parseFloat(newBudget.amount),
      lastUpdated: 'Just now'
    };

    setBudgets([...budgets, budget]);
    setShowCreateBudget(false);
    setNewBudget({
      name: '',
      type: 'monthly',
      amount: '',
      department: '',
      owner: '',
      alertThresholds: { warning: 80, critical: 95 }
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <DollarSign className="h-12 w-12 animate-pulse mx-auto mb-4" />
          <p className="text-muted-foreground">Loading budget management...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Budget Management</h1>
          <p className="text-muted-foreground">
            Track budgets, set alerts, and prevent cost overruns
          </p>
        </div>
        <div className="flex gap-2">
          <Dialog open={showCreateBudget} onOpenChange={setShowCreateBudget}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Create Budget
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Create New Budget</DialogTitle>
                <DialogDescription>
                  Set up a new budget with spending limits and alert thresholds
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Budget Name</Label>
                    <Input
                      id="name"
                      value={newBudget.name}
                      onChange={(e) => setNewBudget({ ...newBudget, name: e.target.value })}
                      placeholder="e.g., Q1 Cloud Infrastructure"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="type">Budget Type</Label>
                    <Select
                      value={newBudget.type}
                      onValueChange={(value) => setNewBudget({ ...newBudget, type: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="annual">Annual</SelectItem>
                        <SelectItem value="quarterly">Quarterly</SelectItem>
                        <SelectItem value="monthly">Monthly</SelectItem>
                        <SelectItem value="project">Project</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="amount">Budget Amount</Label>
                    <Input
                      id="amount"
                      type="number"
                      value={newBudget.amount}
                      onChange={(e) => setNewBudget({ ...newBudget, amount: e.target.value })}
                      placeholder="100000"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="department">Department</Label>
                    <Input
                      id="department"
                      value={newBudget.department}
                      onChange={(e) => setNewBudget({ ...newBudget, department: e.target.value })}
                      placeholder="Engineering"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="owner">Budget Owner</Label>
                    <Input
                      id="owner"
                      value={newBudget.owner}
                      onChange={(e) => setNewBudget({ ...newBudget, owner: e.target.value })}
                      placeholder="John Smith"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Alert Thresholds</Label>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Warning at</span>
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            value={newBudget.alertThresholds.warning}
                            onChange={(e) => setNewBudget({
                              ...newBudget,
                              alertThresholds: {
                                ...newBudget.alertThresholds,
                                warning: parseInt(e.target.value)
                              }
                            })}
                            className="w-20"
                          />
                          <span>%</span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Critical at</span>
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            value={newBudget.alertThresholds.critical}
                            onChange={(e) => setNewBudget({
                              ...newBudget,
                              alertThresholds: {
                                ...newBudget.alertThresholds,
                                critical: parseInt(e.target.value)
                              }
                            })}
                            className="w-20"
                          />
                          <span>%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowCreateBudget(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateBudget}>Create Budget</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          <Button variant="outline">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Active Alerts */}
      {budgets.some(b => b.alerts.some(a => a.triggered)) && (
        <Alert className="border-red-500 bg-red-50 dark:bg-red-950/20">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription>
            <div className="font-semibold mb-2">Active Budget Alerts</div>
            <div className="space-y-2">
              {budgets
                .filter(b => b.alerts.some(a => a.triggered))
                .map(budget => (
                  <div key={budget.id} className="flex justify-between items-center">
                    <div>
                      <span className="font-medium">{budget.name}</span>
                      <span className="text-sm text-muted-foreground ml-2">
                        ({budget.percentage.toFixed(1)}% utilized)
                      </span>
                    </div>
                    <Badge variant="destructive">
                      {budget.status === 'exceeded' ? 'EXCEEDED' : 'AT RISK'}
                    </Badge>
                  </div>
                ))}
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* Budget Overview Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Budgeted
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(budgets.reduce((sum, b) => sum + b.amount, 0))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Across {budgets.length} budgets
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Spent
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(budgets.reduce((sum, b) => sum + b.spent, 0))}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <ArrowUpRight className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-500">+12.5% vs last period</span>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Remaining Budget
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(budgets.reduce((sum, b) => sum + b.remaining, 0))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Available to spend
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              At Risk / Exceeded
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {budgets.filter(b => b.status !== 'on-track').length}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Budgets need attention
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="budgets">Budgets</TabsTrigger>
          <TabsTrigger value="alerts">Alerts & Rules</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="departments">Departments</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Budget Utilization */}
            <Card>
              <CardHeader>
                <CardTitle>Budget Utilization by Type</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="10%"
                    outerRadius="80%"
                    data={utilizationByType}
                  >
                    <PolarAngleAxis
                      type="number"
                      domain={[0, 100]}
                      angleAxisId={0}
                      tick={false}
                    />
                    <RadialBar
                      dataKey="value"
                      cornerRadius={10}
                      fill="#8884d8"
                      label={{ position: 'insideStart', fill: '#fff' }}
                    />
                    <Legend />
                  </RadialBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Spending Trend */}
            <Card>
              <CardHeader>
                <CardTitle>30-Day Spending Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={spendingTrend.slice(-7)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis tickFormatter={(value) => `$${value/1000}k`} />
                    <Tooltip formatter={(value: number) => formatCurrency(value)} />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="actual"
                      stackId="1"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.6}
                      name="Actual"
                    />
                    <Line
                      type="monotone"
                      dataKey="budget"
                      stroke="#82ca9d"
                      strokeDasharray="5 5"
                      name="Budget"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Top Budgets */}
          <Card>
            <CardHeader>
              <CardTitle>Budget Status Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {budgets.map(budget => (
                  <div key={budget.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div>
                          <div className="font-medium">{budget.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {budget.department} • {budget.owner}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">
                          {formatCurrency(budget.spent)} / {formatCurrency(budget.amount)}
                        </div>
                        <Badge className={getStatusColor(budget.status)}>
                          {budget.status.replace('-', ' ').toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>{budget.percentage.toFixed(1)}% utilized</span>
                        <span className="text-muted-foreground">
                          {formatCurrency(budget.remaining)} remaining
                        </span>
                      </div>
                      <Progress value={budget.percentage} className="h-2" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="budgets" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>All Budgets</CardTitle>
              <p className="text-sm text-muted-foreground">
                Manage and monitor all active budgets
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Budget Name</th>
                      <th className="text-left p-2">Type</th>
                      <th className="text-left p-2">Period</th>
                      <th className="text-right p-2">Budget</th>
                      <th className="text-right p-2">Spent</th>
                      <th className="text-right p-2">Remaining</th>
                      <th className="text-center p-2">Status</th>
                      <th className="text-center p-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {budgets.map(budget => (
                      <tr key={budget.id} className="border-b hover:bg-muted/50">
                        <td className="p-2">
                          <div>
                            <div className="font-medium">{budget.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {budget.department} • {budget.owner}
                            </div>
                          </div>
                        </td>
                        <td className="p-2">
                          <Badge variant="outline">
                            {budget.type}
                          </Badge>
                        </td>
                        <td className="p-2">{budget.period}</td>
                        <td className="p-2 text-right font-medium">
                          {formatCurrency(budget.amount)}
                        </td>
                        <td className="p-2 text-right">
                          {formatCurrency(budget.spent)}
                        </td>
                        <td className={`p-2 text-right ${budget.remaining < 0 ? 'text-red-600' : ''}`}>
                          {formatCurrency(budget.remaining)}
                        </td>
                        <td className="p-2 text-center">
                          <Badge className={getStatusColor(budget.status)}>
                            {budget.status.replace('-', ' ')}
                          </Badge>
                        </td>
                        <td className="p-2">
                          <div className="flex justify-center gap-1">
                            <Button variant="ghost" size="sm">
                              <Edit className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="sm">
                              <Bell className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="sm">
                              <Trash2 className="h-4 w-4" />
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
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {/* Alert Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Alert Configuration</CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure budget alerts and notification channels
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label>Alert Channels</Label>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Mail className="h-4 w-4" />
                          <span className="text-sm">Email</span>
                        </div>
                        <Switch defaultChecked />
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Slack className="h-4 w-4" />
                          <span className="text-sm">Slack</span>
                        </div>
                        <Switch defaultChecked />
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <BellRing className="h-4 w-4" />
                          <span className="text-sm">PagerDuty</span>
                        </div>
                        <Switch />
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Alert Frequency</Label>
                    <Select defaultValue="realtime">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="realtime">Real-time</SelectItem>
                        <SelectItem value="hourly">Hourly</SelectItem>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Alert Recipients</Label>
                    <Input placeholder="team@company.com" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Automation Rules */}
          <Card>
            <CardHeader>
              <CardTitle>Automation Rules</CardTitle>
              <p className="text-sm text-muted-foreground">
                Automated actions triggered by budget conditions
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {rules.map(rule => (
                  <div key={rule.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <div className="font-medium">{rule.name}</div>
                          <Switch checked={rule.enabled} />
                        </div>
                        <div className="text-sm text-muted-foreground mt-1">
                          <div>Condition: {rule.condition}</div>
                          <div>Action: {rule.action}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        {rule.lastTriggered && (
                          <div className="text-sm">
                            <div className="text-muted-foreground">Last triggered</div>
                            <div>{rule.lastTriggered}</div>
                            <div className="text-xs text-muted-foreground">
                              {rule.triggerCount} times total
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Spending Trends & Forecasting</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={spendingTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis tickFormatter={(value) => `$${value/1000}k`} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#8884d8"
                    strokeWidth={2}
                    name="Actual Spend"
                  />
                  <Line
                    type="monotone"
                    dataKey="budget"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Budget Limit"
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#ffc658"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    name="Forecast"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="departments" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Department Budget Utilization</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={departmentSpending} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={(value) => `$${value/1000}k`} />
                  <YAxis dataKey="department" type="category" width={100} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Legend />
                  <Bar dataKey="spent" fill="#8884d8" name="Spent" />
                  <Bar dataKey="budget" fill="#82ca9d" name="Budget" opacity={0.3} />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2">
                {departmentSpending.map(dept => (
                  <div key={dept.department} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">{dept.department}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-sm text-muted-foreground">
                        {formatCurrency(dept.spent)} / {formatCurrency(dept.budget)}
                      </div>
                      <Badge className={
                        dept.percentage > 90 ? 'bg-red-100 text-red-700 dark:bg-red-900/50' :
                        dept.percentage > 75 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50' :
                        'bg-green-100 text-green-700 dark:bg-green-900/50'
                      }>
                        {dept.percentage.toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BudgetManagementPage;