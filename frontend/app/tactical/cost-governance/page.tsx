'use client';

import React, { useState, useEffect } from 'react';
import { 
  DollarSign, TrendingUp, TrendingDown, AlertTriangle, PieChart, BarChart3,
  LineChart, Target, Bell, Calendar, Clock, Timer, Download, Upload,
  Filter, Search, Settings, Shield, Activity, Eye, Lock, RefreshCw,
  ChevronRight, ChevronDown, MoreVertical, ExternalLink, Info,
  Briefcase, CreditCard, Wallet, Receipt, Calculator, FileText,
  AlertCircle, CheckCircle, XCircle, ArrowUpRight, ArrowDownRight
} from 'lucide-react';

interface Budget {
  id: string;
  name: string;
  type: 'department' | 'project' | 'resource' | 'subscription';
  allocated: number;
  spent: number;
  forecast: number;
  period: 'monthly' | 'quarterly' | 'annual';
  status: 'on-track' | 'warning' | 'exceeded';
  owner: string;
  alerts: Alert[];
  tags: string[];
}

interface CostAnomaly {
  id: string;
  resource: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  detected: string;
  amount: number;
  percentChange: number;
  baseline: number;
  recommendation: string;
  status: 'new' | 'investigating' | 'resolved' | 'ignored';
}

interface CostPolicy {
  id: string;
  name: string;
  type: 'tagging' | 'budget' | 'resource' | 'optimization';
  scope: string[];
  rules: Rule[];
  violations: number;
  savings: number;
  enforcement: 'audit' | 'warn' | 'enforce';
  status: 'active' | 'draft' | 'disabled';
}

interface Rule {
  id: string;
  condition: string;
  action: string;
  threshold: number;
}

interface Alert {
  id: string;
  type: 'budget' | 'anomaly' | 'policy';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: string;
}

interface CostAllocation {
  id: string;
  name: string;
  type: 'department' | 'project' | 'service' | 'environment';
  cost: number;
  percentage: number;
  trend: 'up' | 'down' | 'stable';
  resources: number;
  children?: CostAllocation[];
}

export default function CostGovernance() {
  const [budgets, setBudgets] = useState<Budget[]>([]);
  const [anomalies, setAnomalies] = useState<CostAnomaly[]>([]);
  const [policies, setPolicies] = useState<CostPolicy[]>([]);
  const [allocations, setAllocations] = useState<CostAllocation[]>([]);
  const [viewMode, setViewMode] = useState<'overview' | 'budgets' | 'policies' | 'anomalies' | 'allocation'>('overview');
  const [timeRange, setTimeRange] = useState('30d');
  const [expandedBudget, setExpandedBudget] = useState<string | null>(null);
  const [selectedDepartment, setSelectedDepartment] = useState('all');

  useEffect(() => {
    // Initialize with cost governance data
    setBudgets([
      {
        id: 'BUD-001',
        name: 'Engineering Department',
        type: 'department',
        allocated: 150000,
        spent: 125000,
        forecast: 145000,
        period: 'monthly',
        status: 'warning',
        owner: 'CTO',
        alerts: [
          { id: 'A1', type: 'budget', severity: 'warning', message: 'Approaching 85% of budget', timestamp: '2 hours ago' }
        ],
        tags: ['engineering', 'cloud', 'infrastructure']
      },
      {
        id: 'BUD-002',
        name: 'Marketing Campaign Q4',
        type: 'project',
        allocated: 50000,
        spent: 32000,
        forecast: 48000,
        period: 'quarterly',
        status: 'on-track',
        owner: 'CMO',
        alerts: [],
        tags: ['marketing', 'campaigns', 'q4']
      },
      {
        id: 'BUD-003',
        name: 'Production Environment',
        type: 'resource',
        allocated: 200000,
        spent: 185000,
        forecast: 210000,
        period: 'monthly',
        status: 'exceeded',
        owner: 'DevOps Lead',
        alerts: [
          { id: 'A2', type: 'budget', severity: 'critical', message: 'Budget exceeded by 5%', timestamp: '30 minutes ago' },
          { id: 'A3', type: 'anomaly', severity: 'warning', message: 'Unusual spike in compute costs', timestamp: '1 hour ago' }
        ],
        tags: ['production', 'azure', 'critical']
      },
      {
        id: 'BUD-004',
        name: 'Data Analytics Platform',
        type: 'project',
        allocated: 75000,
        spent: 45000,
        forecast: 70000,
        period: 'quarterly',
        status: 'on-track',
        owner: 'Data Team Lead',
        alerts: [],
        tags: ['analytics', 'data', 'ai']
      },
      {
        id: 'BUD-005',
        name: 'Development Subscription',
        type: 'subscription',
        allocated: 30000,
        spent: 28500,
        forecast: 31000,
        period: 'monthly',
        status: 'warning',
        owner: 'Dev Manager',
        alerts: [
          { id: 'A4', type: 'budget', severity: 'warning', message: '95% of budget consumed', timestamp: '4 hours ago' }
        ],
        tags: ['development', 'testing', 'sandbox']
      }
    ]);

    setAnomalies([
      {
        id: 'ANOM-001',
        resource: 'VM-PROD-WEB-01',
        type: 'Compute',
        severity: 'high',
        detected: '2 hours ago',
        amount: 2500,
        percentChange: 250,
        baseline: 1000,
        recommendation: 'Review VM size and usage patterns',
        status: 'investigating'
      },
      {
        id: 'ANOM-002',
        resource: 'Storage Account SA-BACKUP',
        type: 'Storage',
        severity: 'medium',
        detected: '5 hours ago',
        amount: 800,
        percentChange: 160,
        baseline: 500,
        recommendation: 'Check for duplicate backups',
        status: 'new'
      },
      {
        id: 'ANOM-003',
        resource: 'SQL-DB-ANALYTICS',
        type: 'Database',
        severity: 'critical',
        detected: '30 minutes ago',
        amount: 5000,
        percentChange: 400,
        baseline: 1250,
        recommendation: 'Investigate query performance and DTU usage',
        status: 'new'
      },
      {
        id: 'ANOM-004',
        resource: 'CDN-GLOBAL',
        type: 'Networking',
        severity: 'low',
        detected: '1 day ago',
        amount: 300,
        percentChange: 50,
        baseline: 200,
        recommendation: 'Review CDN caching policies',
        status: 'resolved'
      }
    ]);

    setPolicies([
      {
        id: 'POL-001',
        name: 'Mandatory Resource Tagging',
        type: 'tagging',
        scope: ['All Subscriptions'],
        rules: [
          { id: 'R1', condition: 'Resource without tags', action: 'Deny creation', threshold: 0 }
        ],
        violations: 45,
        savings: 12000,
        enforcement: 'enforce',
        status: 'active'
      },
      {
        id: 'POL-002',
        name: 'Auto-shutdown Non-Prod VMs',
        type: 'optimization',
        scope: ['Dev', 'Test', 'Staging'],
        rules: [
          { id: 'R2', condition: 'VM idle > 2 hours', action: 'Auto-shutdown', threshold: 2 }
        ],
        violations: 12,
        savings: 35000,
        enforcement: 'enforce',
        status: 'active'
      },
      {
        id: 'POL-003',
        name: 'Budget Alert Thresholds',
        type: 'budget',
        scope: ['All Departments'],
        rules: [
          { id: 'R3', condition: 'Budget > 80%', action: 'Send alert', threshold: 80 },
          { id: 'R4', condition: 'Budget > 90%', action: 'Escalate to manager', threshold: 90 }
        ],
        violations: 8,
        savings: 0,
        enforcement: 'warn',
        status: 'active'
      },
      {
        id: 'POL-004',
        name: 'Reserved Instance Optimization',
        type: 'optimization',
        scope: ['Production'],
        rules: [
          { id: 'R5', condition: 'VM usage > 70%', action: 'Recommend RI', threshold: 70 }
        ],
        violations: 0,
        savings: 85000,
        enforcement: 'audit',
        status: 'active'
      }
    ]);

    setAllocations([
      {
        id: 'ALLOC-001',
        name: 'Engineering',
        type: 'department',
        cost: 450000,
        percentage: 35,
        trend: 'up',
        resources: 1250,
        children: [
          { id: 'ALLOC-001-1', name: 'Infrastructure', type: 'service', cost: 200000, percentage: 44, trend: 'stable', resources: 450 },
          { id: 'ALLOC-001-2', name: 'Development', type: 'service', cost: 150000, percentage: 33, trend: 'up', resources: 500 },
          { id: 'ALLOC-001-3', name: 'QA/Testing', type: 'service', cost: 100000, percentage: 23, trend: 'down', resources: 300 }
        ]
      },
      {
        id: 'ALLOC-002',
        name: 'Operations',
        type: 'department',
        cost: 380000,
        percentage: 30,
        trend: 'stable',
        resources: 980
      },
      {
        id: 'ALLOC-003',
        name: 'Sales & Marketing',
        type: 'department',
        cost: 250000,
        percentage: 20,
        trend: 'down',
        resources: 620
      },
      {
        id: 'ALLOC-004',
        name: 'Data & Analytics',
        type: 'department',
        cost: 190000,
        percentage: 15,
        trend: 'up',
        resources: 420
      }
    ]);
  }, []);

  const totalBudget = budgets.reduce((acc, b) => acc + b.allocated, 0);
  const totalSpent = budgets.reduce((acc, b) => acc + b.spent, 0);
  const totalForecast = budgets.reduce((acc, b) => acc + b.forecast, 0);
  const totalSavings = policies.reduce((acc, p) => acc + p.savings, 0);
  const activeAnomalies = anomalies.filter(a => a.status !== 'resolved').length;
  const policyViolations = policies.reduce((acc, p) => acc + p.violations, 0);

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'on-track': case 'active': case 'resolved': return 'text-green-500 bg-green-900/20';
      case 'warning': case 'investigating': return 'text-yellow-500 bg-yellow-900/20';
      case 'exceeded': case 'critical': case 'new': return 'text-red-500 bg-red-900/20';
      case 'draft': case 'ignored': return 'text-gray-500 bg-gray-900/20';
      default: return 'text-gray-500 bg-gray-900/20';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch(severity) {
      case 'low': return 'text-blue-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-orange-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold flex items-center space-x-2">
              <DollarSign className="w-6 h-6 text-green-500" />
              <span>Cost Governance</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Budget management, cost optimization, and financial policies</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
            >
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last Quarter</option>
              <option value="1y">Last Year</option>
            </select>

            <button className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm flex items-center space-x-2">
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>

            <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm flex items-center space-x-2">
              <Calculator className="w-4 h-4" />
              <span>Cost Analysis</span>
            </button>
          </div>
        </div>
      </header>

      {/* Summary Stats */}
      <div className="bg-gray-900/50 border-b border-gray-800 px-6 py-3">
        <div className="grid grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-xs text-gray-500">Total Budget</div>
            <div className="text-2xl font-bold">${(totalBudget/1000).toFixed(0)}K</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Spent</div>
            <div className="text-2xl font-bold text-yellow-500">${(totalSpent/1000).toFixed(0)}K</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Forecast</div>
            <div className="text-2xl font-bold text-blue-500">${(totalForecast/1000).toFixed(0)}K</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Savings</div>
            <div className="text-2xl font-bold text-green-500">${(totalSavings/1000).toFixed(0)}K</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Anomalies</div>
            <div className="text-2xl font-bold text-orange-500">{activeAnomalies}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">Violations</div>
            <div className="text-2xl font-bold text-red-500">{policyViolations}</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-gray-900/30 border-b border-gray-800 px-6">
        <div className="flex space-x-6">
          {['overview', 'budgets', 'policies', 'anomalies', 'allocation'].map(view => (
            <button
              key={view}
              onClick={() => setViewMode(view as any)}
              className={`py-3 border-b-2 text-sm capitalize ${
                viewMode === view 
                  ? 'border-green-500 text-green-500' 
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              {view === 'allocation' ? 'Cost Allocation' : view}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {viewMode === 'overview' && (
          <div className="space-y-6">
            {/* Budget Overview */}
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h2 className="text-sm font-bold mb-4">Budget Utilization</h2>
                <div className="space-y-3">
                  {budgets.slice(0, 3).map(budget => (
                    <div key={budget.id}>
                      <div className="flex justify-between text-xs mb-1">
                        <span>{budget.name}</span>
                        <span>{((budget.spent/budget.allocated)*100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${
                            budget.status === 'exceeded' ? 'bg-red-500' :
                            budget.status === 'warning' ? 'bg-yellow-500' :
                            'bg-green-500'
                          }`}
                          style={{ width: `${(budget.spent/budget.allocated)*100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h2 className="text-sm font-bold mb-4">Cost Trend</h2>
                <div className="h-32 flex items-center justify-center">
                  <LineChart className="w-24 h-24 text-gray-600" />
                </div>
                <p className="text-xs text-gray-500 text-center">Visualization coming soon</p>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <h2 className="text-sm font-bold mb-4">Recent Alerts</h2>
                <div className="space-y-2">
                  {budgets.flatMap(b => b.alerts).slice(0, 3).map(alert => (
                    <div key={alert.id} className="flex items-start space-x-2">
                      <AlertTriangle className={`w-4 h-4 ${
                        alert.severity === 'critical' ? 'text-red-500' :
                        alert.severity === 'warning' ? 'text-yellow-500' :
                        'text-blue-500'
                      }`} />
                      <div className="flex-1">
                        <p className="text-xs">{alert.message}</p>
                        <p className="text-xs text-gray-500">{alert.timestamp}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Cost Savings Opportunities */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h2 className="text-sm font-bold mb-4">Top Cost Savings Opportunities</h2>
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-gray-800 rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">Reserved Instances</span>
                    <ArrowUpRight className="w-4 h-4 text-green-500" />
                  </div>
                  <div className="text-xl font-bold text-green-500">$85K</div>
                  <p className="text-xs text-gray-500">Annual savings potential</p>
                </div>
                <div className="bg-gray-800 rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">Idle Resources</span>
                    <ArrowUpRight className="w-4 h-4 text-yellow-500" />
                  </div>
                  <div className="text-xl font-bold text-yellow-500">$35K</div>
                  <p className="text-xs text-gray-500">Monthly savings</p>
                </div>
                <div className="bg-gray-800 rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">Right-sizing</span>
                    <ArrowUpRight className="w-4 h-4 text-blue-500" />
                  </div>
                  <div className="text-xl font-bold text-blue-500">$52K</div>
                  <p className="text-xs text-gray-500">Quarterly savings</p>
                </div>
                <div className="bg-gray-800 rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">Spot Instances</span>
                    <ArrowUpRight className="w-4 h-4 text-purple-500" />
                  </div>
                  <div className="text-xl font-bold text-purple-500">$28K</div>
                  <p className="text-xs text-gray-500">Monthly potential</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'budgets' && (
          <div className="space-y-4">
            {budgets.map(budget => (
              <div key={budget.id} className="bg-gray-900 border border-gray-800 rounded-lg">
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-sm font-bold">{budget.name}</h3>
                        <span className={`px-2 py-1 rounded text-xs ${getStatusColor(budget.status)}`}>
                          {budget.status}
                        </span>
                        <span className="px-2 py-1 bg-gray-800 rounded text-xs">
                          {budget.period}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Owner: {budget.owner}</span>
                        <span>Type: {budget.type}</span>
                      </div>
                    </div>
                    <button 
                      onClick={() => setExpandedBudget(expandedBudget === budget.id ? null : budget.id)}
                      className="p-1 hover:bg-gray-800 rounded"
                    >
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${
                        expandedBudget === budget.id ? 'rotate-180' : ''
                      }`} />
                    </button>
                  </div>

                  <div className="grid grid-cols-4 gap-4 mb-3">
                    <div>
                      <div className="text-xs text-gray-500">Allocated</div>
                      <div className="text-xl font-bold">${(budget.allocated/1000).toFixed(0)}K</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Spent</div>
                      <div className={`text-xl font-bold ${
                        budget.spent > budget.allocated ? 'text-red-500' :
                        budget.spent > budget.allocated * 0.8 ? 'text-yellow-500' :
                        'text-green-500'
                      }`}>
                        ${(budget.spent/1000).toFixed(0)}K
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Forecast</div>
                      <div className="text-xl font-bold text-blue-500">
                        ${(budget.forecast/1000).toFixed(0)}K
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Utilization</div>
                      <div className="text-xl font-bold">
                        {((budget.spent/budget.allocated)*100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden mb-3">
                    <div className="h-full flex">
                      <div 
                        className="bg-green-500"
                        style={{ width: `${Math.min((budget.spent/budget.allocated)*100, 100)}%` }}
                      />
                      {budget.forecast > budget.spent && (
                        <div 
                          className="bg-blue-500 opacity-50"
                          style={{ width: `${Math.min(((budget.forecast - budget.spent)/budget.allocated)*100, 100 - (budget.spent/budget.allocated)*100)}%` }}
                        />
                      )}
                    </div>
                  </div>

                  {expandedBudget === budget.id && (
                    <div className="mt-4 pt-4 border-t border-gray-800">
                      {budget.alerts.length > 0 && (
                        <div className="mb-4">
                          <h4 className="text-xs font-bold mb-2">Active Alerts</h4>
                          <div className="space-y-2">
                            {budget.alerts.map(alert => (
                              <div key={alert.id} className="flex items-center justify-between bg-gray-800 rounded p-2">
                                <div className="flex items-center space-x-2">
                                  <Bell className={`w-4 h-4 ${getSeverityColor(alert.severity)}`} />
                                  <span className="text-xs">{alert.message}</span>
                                </div>
                                <span className="text-xs text-gray-500">{alert.timestamp}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="flex items-center space-x-2">
                        {budget.tags.map(tag => (
                          <span key={tag} className="px-2 py-1 bg-gray-800 rounded text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'policies' && (
          <div className="space-y-4">
            {policies.map(policy => (
              <div key={policy.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <h3 className="text-sm font-bold">{policy.name}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(policy.status)}`}>
                        {policy.status}
                      </span>
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs">
                        {policy.enforcement}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">Type: {policy.type}</div>
                  </div>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-3">
                  <div>
                    <div className="text-xs text-gray-500">Rules</div>
                    <div className="text-lg font-bold">{policy.rules.length}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Violations</div>
                    <div className={`text-lg font-bold ${
                      policy.violations > 0 ? 'text-orange-500' : 'text-green-500'
                    }`}>
                      {policy.violations}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Savings</div>
                    <div className="text-lg font-bold text-green-500">
                      ${(policy.savings/1000).toFixed(0)}K
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Scope</div>
                    <div className="text-lg font-bold">{policy.scope.length}</div>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                    View Details
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Edit Policy
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    View Violations
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'anomalies' && (
          <div className="space-y-4">
            {anomalies.map(anomaly => (
              <div key={anomaly.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <h3 className="text-sm font-bold">{anomaly.resource}</h3>
                      <span className={`px-2 py-1 rounded text-xs ${getStatusColor(anomaly.status)}`}>
                        {anomaly.status}
                      </span>
                      <span className={`px-2 py-1 bg-gray-800 rounded text-xs ${getSeverityColor(anomaly.severity)}`}>
                        {anomaly.severity}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      Type: {anomaly.type} | Detected: {anomaly.detected}
                    </div>
                  </div>
                  <button className="p-1 hover:bg-gray-800 rounded">
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-3">
                  <div>
                    <div className="text-xs text-gray-500">Current Cost</div>
                    <div className="text-lg font-bold text-red-500">${anomaly.amount}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Baseline</div>
                    <div className="text-lg font-bold">${anomaly.baseline}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Change</div>
                    <div className={`text-lg font-bold ${
                      anomaly.percentChange > 100 ? 'text-red-500' : 'text-orange-500'
                    }`}>
                      +{anomaly.percentChange}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Impact</div>
                    <div className="text-lg font-bold">
                      ${anomaly.amount - anomaly.baseline}
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800 rounded p-2 mb-3">
                  <p className="text-xs text-gray-400">Recommendation:</p>
                  <p className="text-xs">{anomaly.recommendation}</p>
                </div>

                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs">
                    Investigate
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    View Resource
                  </button>
                  <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs">
                    Ignore
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {viewMode === 'allocation' && (
          <div className="space-y-4">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <h2 className="text-sm font-bold mb-4">Cost Allocation by Department</h2>
              <div className="space-y-3">
                {allocations.map(allocation => (
                  <div key={allocation.id}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Building className="w-4 h-4 text-gray-500" />
                        <span className="text-sm font-medium">{allocation.name}</span>
                        {allocation.trend === 'up' ? (
                          <TrendingUp className="w-4 h-4 text-red-500" />
                        ) : allocation.trend === 'down' ? (
                          <TrendingDown className="w-4 h-4 text-green-500" />
                        ) : (
                          <Activity className="w-4 h-4 text-gray-500" />
                        )}
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className="text-sm">${(allocation.cost/1000).toFixed(0)}K</span>
                        <span className="text-sm text-gray-500">{allocation.percentage}%</span>
                        <span className="text-xs text-gray-500">{allocation.resources} resources</span>
                      </div>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-600 to-blue-400"
                        style={{ width: `${allocation.percentage}%` }}
                      />
                    </div>
                    {allocation.children && (
                      <div className="ml-6 mt-2 space-y-2">
                        {allocation.children.map(child => (
                          <div key={child.id} className="flex items-center justify-between text-xs">
                            <span className="text-gray-400">{child.name}</span>
                            <div className="flex items-center space-x-2">
                              <span>${(child.cost/1000).toFixed(0)}K</span>
                              <span className="text-gray-500">({child.percentage}%)</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}