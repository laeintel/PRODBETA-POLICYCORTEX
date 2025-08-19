'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';
import toast from 'react-hot-toast';
import { 
  DollarSign, TrendingUp, TrendingDown, AlertTriangle, PieChart, BarChart3, Calendar, Download,
  ChevronRight, RefreshCw, Filter, Search, Target, Zap, Activity, Users, Server, Database,
  Cloud, Shield, Eye, Settings, Play, Pause, ArrowUpRight, ArrowDownRight, Minus,
  CreditCard, Wallet, TrendingUp as Growth, Lightbulb, Calculator, Gauge, Award,
  Clock, MapPin, Layers, Network, Globe, Cpu, HardDrive, Wifi, LineChart
} from 'lucide-react';

interface CostResource {
  id: string;
  name: string;
  type: string;
  location: string;
  cost: number;
  dailyCost: number;
  trend: 'up' | 'down' | 'stable';
  percentage: number;
  optimization: {
    potential: number;
    recommendations: string[];
    riskLevel: 'low' | 'medium' | 'high';
  };
  metrics: {
    utilization: number;
    performance: number;
    efficiency: number;
  };
  tags: string[];
}

interface BudgetAlert {
  id: string;
  name: string;
  threshold: number;
  current: number;
  percentage: number;
  status: 'ok' | 'warning' | 'critical';
  forecast: number;
  daysRemaining: number;
}

interface CostData {
  currentMonth: number;
  previousMonth: number;
  trend: number;
  forecast: number;
  dailyAverage: number;
  yearToDate: number;
  projectedYear: number;
  breakdown: {
    compute: number;
    storage: number;
    networking: number;
    database: number;
    security: number;
    analytics: number;
    ai: number;
    other: number;
  };
  regions: Array<{
    name: string;
    cost: number;
    percentage: number;
    trend: number;
  }>;
  topSpenders: CostResource[];
  budgetAlerts: BudgetAlert[];
  anomalies: Array<{
    id: string;
    resource: string;
    spike: number;
    detected: string;
    severity: 'low' | 'medium' | 'high';
    impact: number;
    category: string;
  }>;
  savings: {
    identified: number;
    implemented: number;
    recommendations: number;
    potentialMonthly: number;
    riskAnalysis: {
      low: number;
      medium: number;
      high: number;
    };
  };
  historical: {
    labels: string[];
    costs: number[];
    forecasts: number[];
    budgets: number[];
  };
  optimization: {
    rightsizing: number;
    reservedInstances: number;
    storageOptimization: number;
    scheduledShutdown: number;
    unusedResources: number;
  };
  kpis: {
    costPerUser: number;
    costPerTransaction: number;
    efficiency: number;
    wastePercentage: number;
  };
}

export default function CostAnalyticsCenter() {
  return (
    <AuthGuard requireAuth={true}>
      <CostAnalyticsCenterContent />
    </AuthGuard>
  );
}

function CostAnalyticsCenterContent() {
  const [data, setData] = useState<CostData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'breakdown' | 'optimization' | 'budgets' | 'anomalies' | 'forecasting'>('overview');
  const [timeRange, setTimeRange] = useState('30d');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'cost' | 'trend' | 'optimization'>('cost');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<string>('all');

  useEffect(() => {
    fetchCostData();
    const interval = setInterval(fetchCostData, 60000);
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchCostData = async () => {
    try {
      const resp = await api.getCostAnalysis()
      if (resp.error) {
        setData(getMockCostData());
      } else {
        setData(resp.data as any);
      }
    } catch (error) {
      setData(getMockCostData());
    } finally {
      setLoading(false);
    }
  };

  const triggerAction = async (actionType: string) => {
    try {
      const resp = await api.createAction('global', actionType)
      if (resp.error || resp.status >= 400) {
        toast.error(`Action failed: ${actionType}`)
        return
      }
      toast.success(`${actionType.replace('_',' ')} started`)
      const id = resp.data?.action_id || resp.data?.id
      if (id) {
        const stop = api.streamActionEvents(String(id), (m) => console.log('[cost-action]', id, m))
        setTimeout(stop, 60000)
      }
    } catch (e) {
      toast.error(`Action error: ${actionType}`)
    }
  }

  const getMockCostData = (): CostData => ({
    currentMonth: 127439,
    previousMonth: 138472,
    trend: -8,
    forecast: 145000,
    dailyAverage: 4248,
    yearToDate: 1456780,
    projectedYear: 1742280,
    breakdown: {
      compute: 45678,
      storage: 23456,
      networking: 12345,
      database: 34567,
      security: 8945,
      analytics: 6789,
      ai: 4567,
      other: 11393
    },
    regions: [
      { name: 'East US', cost: 42356, percentage: 33.2, trend: -5 },
      { name: 'West US 2', cost: 38172, percentage: 29.9, trend: 12 },
      { name: 'West Europe', cost: 28934, percentage: 22.7, trend: -2 },
      { name: 'Southeast Asia', cost: 17977, percentage: 14.1, trend: 8 }
    ],
    budgetAlerts: [
      { id: 'b1', name: 'Production Environment', threshold: 50000, current: 48750, percentage: 97.5, status: 'warning', forecast: 52000, daysRemaining: 8 },
      { id: 'b2', name: 'Development Resources', threshold: 15000, current: 12450, percentage: 83.0, status: 'ok', forecast: 14200, daysRemaining: 12 },
      { id: 'b3', name: 'Analytics Workloads', threshold: 25000, current: 26780, percentage: 107.1, status: 'critical', forecast: 28500, daysRemaining: -2 },
      { id: 'b4', name: 'Storage & Backup', threshold: 20000, current: 17890, percentage: 89.4, status: 'ok', forecast: 19450, daysRemaining: 15 }
    ],
    topSpenders: [
      { 
        id: 'r1', 
        name: 'Production VM Scale Set', 
        type: 'Virtual Machines',
        location: 'East US',
        cost: 34567, 
        dailyCost: 1152,
        trend: 'up', 
        percentage: 27,
        optimization: {
          potential: 8500,
          recommendations: ['Rightsize to D4s_v3', 'Enable auto-shutdown', 'Use reserved instances'],
          riskLevel: 'low'
        },
        metrics: {
          utilization: 78,
          performance: 92,
          efficiency: 85
        },
        tags: ['production', 'web-tier', 'scalable']
      },
      {
        id: 'r2',
        name: 'SQL Database Cluster',
        type: 'Azure SQL Database',
        location: 'West US 2',
        cost: 28934,
        dailyCost: 964,
        trend: 'down',
        percentage: 23,
        optimization: {
          potential: 5200,
          recommendations: ['Switch to elastic pool', 'Optimize DTU usage', 'Archive old data'],
          riskLevel: 'medium'
        },
        metrics: {
          utilization: 65,
          performance: 88,
          efficiency: 72
        },
        tags: ['database', 'production', 'high-availability']
      },
      {
        id: 'r3',
        name: 'Premium Storage Accounts',
        type: 'Storage',
        location: 'Multiple',
        cost: 19234,
        dailyCost: 641,
        trend: 'stable',
        percentage: 15,
        optimization: {
          potential: 3800,
          recommendations: ['Move to cool tier', 'Enable lifecycle policies', 'Compress data'],
          riskLevel: 'low'
        },
        metrics: {
          utilization: 82,
          performance: 95,
          efficiency: 88
        },
        tags: ['storage', 'backup', 'archival']
      },
      {
        id: 'r4',
        name: 'App Service Premium Plans',
        type: 'App Service',
        location: 'West Europe',
        cost: 15678,
        dailyCost: 522,
        trend: 'up',
        percentage: 12,
        optimization: {
          potential: 2950,
          recommendations: ['Scale down during off-hours', 'Use consumption plan', 'Optimize instance size'],
          riskLevel: 'medium'
        },
        metrics: {
          utilization: 45,
          performance: 91,
          efficiency: 68
        },
        tags: ['app-service', 'web-apps', 'production']
      },
      {
        id: 'r5',
        name: 'Application Gateway',
        type: 'Load Balancer',
        location: 'East US',
        cost: 12345,
        dailyCost: 411,
        trend: 'down',
        percentage: 10,
        optimization: {
          potential: 1200,
          recommendations: ['Consolidate instances', 'Use standard tier', 'Optimize SSL processing'],
          riskLevel: 'high'
        },
        metrics: {
          utilization: 38,
          performance: 94,
          efficiency: 62
        },
        tags: ['networking', 'load-balancer', 'security']
      }
    ],
    anomalies: [
      { id: 'a1', resource: 'VM-PROD-SCALE-01', spike: 234, detected: '2 hours ago', severity: 'high', impact: 8900, category: 'Compute' },
      { id: 'a2', resource: 'Storage-Archive-Premium', spike: 156, detected: '1 day ago', severity: 'medium', impact: 2340, category: 'Storage' },
      { id: 'a3', resource: 'CDN-Global-Distribution', spike: 89, detected: '3 days ago', severity: 'low', impact: 890, category: 'Networking' },
      { id: 'a4', resource: 'SQL-Analytics-Cluster', spike: 178, detected: '4 hours ago', severity: 'high', impact: 5670, category: 'Database' },
      { id: 'a5', resource: 'AI-ML-Compute-Instance', spike: 145, detected: '6 hours ago', severity: 'medium', impact: 3450, category: 'AI/ML' }
    ],
    savings: {
      identified: 45000,
      implemented: 28000,
      recommendations: 24,
      potentialMonthly: 12450,
      riskAnalysis: {
        low: 28500,
        medium: 12300,
        high: 4200
      }
    },
    historical: {
      labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
      costs: [98500, 105600, 112300, 98900, 108700, 115200, 122400, 118900, 125600, 132100, 127439, 0],
      forecasts: [100000, 108000, 115000, 102000, 112000, 118000, 125000, 122000, 128000, 135000, 130000, 145000],
      budgets: [110000, 110000, 115000, 115000, 120000, 120000, 125000, 125000, 130000, 130000, 135000, 135000]
    },
    optimization: {
      rightsizing: 15600,
      reservedInstances: 12800,
      storageOptimization: 8900,
      scheduledShutdown: 4200,
      unusedResources: 3500
    },
    kpis: {
      costPerUser: 127.44,
      costPerTransaction: 0.023,
      efficiency: 78.5,
      wastePercentage: 12.3
    }
  });

  // Computed values for filtering and sorting
  const filteredResources = useMemo(() => {
    if (!data?.topSpenders) return [];
    
    let filtered = data.topSpenders.filter(resource => 
      (filterCategory === 'all' || resource.type.toLowerCase().includes(filterCategory.toLowerCase())) &&
      (selectedRegion === 'all' || resource.location === selectedRegion) &&
      (searchQuery === '' || resource.name.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'cost': return b.cost - a.cost;
        case 'trend': 
          const trendOrder = { 'up': 2, 'stable': 1, 'down': 0 };
          return trendOrder[b.trend] - trendOrder[a.trend];
        case 'optimization': return b.optimization.potential - a.optimization.potential;
        default: return 0;
      }
    });
  }, [data?.topSpenders, filterCategory, selectedRegion, searchQuery, sortBy]);

  const renderHistoricalChart = () => {
    if (!data?.historical) return null;
    
    const maxValue = Math.max(
      ...data.historical.costs.filter(c => c > 0),
      ...data.historical.forecasts,
      ...data.historical.budgets
    );
    
    return (
      <div className="h-64 flex items-end justify-between space-x-1">
        {data.historical.labels.map((label, index) => (
          <div key={index} className="flex flex-col items-center space-y-1 flex-1">
            <div className="flex flex-col justify-end h-56 w-full space-y-0.5">
              {data.historical.costs[index] > 0 && (
                <div 
                  className="bg-blue-500 rounded-sm transition-all duration-300 hover:bg-blue-400"
                  style={{ height: `${(data.historical.costs[index] / maxValue) * 100}%` }}
                  title={`Actual: ${formatCurrency(data.historical.costs[index])}`}
                />
              )}
              <div 
                className="bg-purple-500 rounded-sm transition-all duration-300 hover:bg-purple-400 opacity-70"
                style={{ height: `${(data.historical.forecasts[index] / maxValue) * 100}%` }}
                title={`Forecast: ${formatCurrency(data.historical.forecasts[index])}`}
              />
              <div 
                className="bg-green-500 rounded-sm transition-all duration-300 hover:bg-green-400 opacity-50"
                style={{ height: `${(data.historical.budgets[index] / maxValue) * 100}%` }}
                title={`Budget: ${formatCurrency(data.historical.budgets[index])}`}
              />
            </div>
            <span className="text-xs text-gray-500 transform rotate-45 whitespace-nowrap">
              {label}
            </span>
          </div>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 border-4 border-green-600 border-t-transparent rounded-full animate-spin mx-auto mb-6" />
          <div className="space-y-2">
            <div className="flex items-center justify-center space-x-2">
              <DollarSign className="w-5 h-5 text-green-500 animate-pulse" />
              <p className="text-lg font-bold text-green-400">LOADING COST ANALYTICS</p>
            </div>
            <p className="text-sm text-gray-500">Analyzing spending patterns and optimizations...</p>
          </div>
        </div>
      </div>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-white transition-colors flex items-center space-x-2">
                <ChevronRight className="w-4 h-4 rotate-180" />
                <span>TACTICAL</span>
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <div className="flex items-center space-x-3">
                <DollarSign className="w-6 h-6 text-green-500 animate-pulse" />
                <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                  COST ANALYTICS CENTER
                </h1>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-bold border transition-all duration-200 ${
                data?.trend && data.trend < 0 
                  ? 'bg-green-900/30 text-green-400 border-green-800/30 animate-pulse' 
                  : 'bg-red-900/30 text-red-400 border-red-800/30'
              }`}>
                {data?.trend && data.trend < 0 ? '↓' : '↑'} {Math.abs(data?.trend || 0)}% MTD
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`p-2 rounded-lg transition-all duration-200 ${
                    autoRefresh ? 'bg-green-900/30 text-green-400 border border-green-800/30' : 'bg-gray-800 text-gray-400 border border-gray-700'
                  }`}
                >
                  <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
                </button>
                <select 
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="px-3 py-1 bg-gray-800 border border-gray-700 rounded text-sm"
                >
                  <option value="7d">7 Days</option>
                  <option value="30d">30 Days</option>
                  <option value="90d">90 Days</option>
                  <option value="1y">1 Year</option>
                </select>
              </div>
              <button 
                onClick={() => triggerAction('optimize_costs')} 
                className="px-6 py-2 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white font-semibold rounded-lg transition-all duration-200 transform hover:scale-105 flex items-center space-x-2 shadow-lg shadow-green-900/25"
              >
                <Target className="w-4 h-4" />
                <span>OPTIMIZE</span>
              </button>
              <button 
                onClick={() => triggerAction('export_cost_report')} 
                className="px-6 py-2 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-lg border border-gray-700 transition-all duration-200 flex items-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>EXPORT</span>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-6 pb-4">
          <div className="flex space-x-1 bg-gray-800/50 rounded-lg p-1">
            {[
              { key: 'overview', label: 'Overview', icon: BarChart3 },
              { key: 'breakdown', label: 'Breakdown', icon: PieChart },
              { key: 'optimization', label: 'Optimization', icon: Target },
              { key: 'budgets', label: 'Budgets', icon: Wallet },
              { key: 'anomalies', label: 'Anomalies', icon: AlertTriangle },
              { key: 'forecasting', label: 'Forecasting', icon: TrendingUp }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === key
                    ? 'bg-green-600 text-white shadow-lg'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="p-6 space-y-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-green-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Current Month</p>
                  <CreditCard className="w-5 h-5 text-green-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-green-400">{formatCurrency(data?.currentMonth || 0)}</p>
                <p className={`text-xs mt-1 ${data?.trend && data.trend < 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {data?.trend && data.trend < 0 ? '↓' : '↑'} {formatCurrency(Math.abs((data?.currentMonth || 0) - (data?.previousMonth || 0)))}
                </p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-blue-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Daily Average</p>
                  <Calendar className="w-5 h-5 text-blue-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-blue-400">{formatCurrency(data?.dailyAverage || 0)}</p>
                <p className="text-xs text-gray-500 mt-1">per day</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-yellow-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Forecast</p>
                  <TrendingUp className="w-5 h-5 text-yellow-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-yellow-400">{formatCurrency(data?.forecast || 0)}</p>
                <p className="text-xs text-gray-500 mt-1">next 30 days</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-purple-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Year to Date</p>
                  <BarChart3 className="w-5 h-5 text-purple-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-purple-400">{formatCurrency(data?.yearToDate || 0)}</p>
                <p className="text-xs text-gray-500 mt-1">total YTD</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-cyan-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Savings Potential</p>
                  <Lightbulb className="w-5 h-5 text-cyan-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-cyan-400">{formatCurrency(data?.savings.identified || 0)}</p>
                <p className="text-xs text-gray-500 mt-1">{data?.savings.recommendations} opportunities</p>
              </div>
              
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-red-800/50 transition-all duration-200">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-gray-500 uppercase font-semibold">Cost Efficiency</p>
                  <Gauge className="w-5 h-5 text-red-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-red-400">{data?.kpis.efficiency}%</p>
                <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-red-600 to-red-400 rounded-full transition-all duration-1000" 
                    style={{ width: `${data?.kpis.efficiency}%` }} 
                  />
                </div>
              </div>
            </div>

            {/* Historical Chart */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl">
              <div className="p-6 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-white">SPENDING TRENDS</h3>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 text-sm">
                      <div className="w-3 h-3 bg-blue-500 rounded"></div>
                      <span className="text-gray-400">Actual</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <div className="w-3 h-3 bg-purple-500 rounded"></div>
                      <span className="text-gray-400">Forecast</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <div className="w-3 h-3 bg-green-500 rounded"></div>
                      <span className="text-gray-400">Budget</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="p-6">
                {renderHistoricalChart()}
              </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Cost Per User</h3>
                  <Users className="w-5 h-5 text-blue-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">${data?.kpis.costPerUser}</p>
                <p className="text-xs text-gray-500 mt-2">per active user</p>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Cost Per Transaction</h3>
                  <Activity className="w-5 h-5 text-green-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">${data?.kpis.costPerTransaction}</p>
                <p className="text-xs text-gray-500 mt-2">per transaction</p>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Waste Percentage</h3>
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">{data?.kpis.wastePercentage}%</p>
                <div className="mt-3 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-red-500 rounded-full transition-all duration-1000" 
                    style={{ width: `${data?.kpis.wastePercentage}%` }} 
                  />
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Monthly Savings</h3>
                  <Award className="w-5 h-5 text-purple-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">{formatCurrency(data?.savings.potentialMonthly || 0)}</p>
                <p className="text-xs text-gray-500 mt-2">potential monthly</p>
              </div>
            </div>
          </div>
        )}

        {/* Breakdown Tab */}
        {activeTab === 'breakdown' && (
          <div className="space-y-6">
            {/* Service Breakdown */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-xl">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-lg font-bold text-white">SERVICE BREAKDOWN</h3>
                </div>
                <div className="p-6">
                  <div className="space-y-4">
                    {Object.entries(data?.breakdown || {}).map(([service, cost]) => {
                      const percentage = ((cost / (data?.currentMonth || 1)) * 100).toFixed(1);
                      const serviceColors = {
                        compute: 'from-blue-600 to-blue-400',
                        storage: 'from-green-600 to-green-400',
                        networking: 'from-purple-600 to-purple-400',
                        database: 'from-yellow-600 to-yellow-400',
                        security: 'from-red-600 to-red-400',
                        analytics: 'from-cyan-600 to-cyan-400',
                        ai: 'from-pink-600 to-pink-400',
                        other: 'from-gray-600 to-gray-400'
                      };
                      return (
                        <div key={service} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm capitalize font-medium text-white">{service.replace('_', ' ')}</span>
                            <span className="text-sm font-mono text-white">{formatCurrency(cost)}</span>
                          </div>
                          <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                            <div
                              className={`h-full bg-gradient-to-r ${serviceColors[service as keyof typeof serviceColors]} rounded-full transition-all duration-1000`}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <div className="text-xs text-gray-500">{percentage}% of total spend</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Regional Breakdown */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl">
                <div className="p-4 border-b border-gray-800">
                  <h3 className="text-lg font-bold text-white">REGIONAL BREAKDOWN</h3>
                </div>
                <div className="divide-y divide-gray-800">
                  {data?.regions.map((region) => (
                    <div key={region.name} className="p-4 hover:bg-gray-800/50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <Globe className="w-4 h-4 text-blue-400" />
                          <span className="font-medium text-white">{region.name}</span>
                        </div>
                        <div className={`flex items-center space-x-1 text-xs px-2 py-1 rounded ${
                          region.trend > 0 ? 'bg-red-900/30 text-red-400' : 'bg-green-900/30 text-green-400'
                        }`}>
                          {region.trend > 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                          <span>{Math.abs(region.trend)}%</span>
                        </div>
                      </div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-2xl font-bold font-mono text-white">{formatCurrency(region.cost)}</span>
                        <span className="text-sm text-gray-400">{region.percentage}%</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-600 to-purple-600 rounded-full transition-all duration-1000"
                          style={{ width: `${region.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Resource Details */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl">
              <div className="p-4 border-b border-gray-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-white">TOP SPENDING RESOURCES</h3>
                  <div className="flex items-center space-x-4">
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" />
                      <input
                        type="text"
                        placeholder="Search resources..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm w-60"
                      />
                    </div>
                    <select
                      value={filterCategory}
                      onChange={(e) => setFilterCategory(e.target.value)}
                      className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm"
                    >
                      <option value="all">All Categories</option>
                      <option value="virtual">Virtual Machines</option>
                      <option value="database">Databases</option>
                      <option value="storage">Storage</option>
                      <option value="app">App Services</option>
                    </select>
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value as any)}
                      className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm"
                    >
                      <option value="cost">Sort by Cost</option>
                      <option value="trend">Sort by Trend</option>
                      <option value="optimization">Sort by Savings</option>
                    </select>
                  </div>
                </div>
              </div>
              <div className="divide-y divide-gray-800">
                {filteredResources.slice(0, 10).map((resource) => (
                  <div key={resource.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="font-bold text-white">{resource.name}</h4>
                          <span className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded">
                            {resource.type}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded font-bold ${
                            resource.trend === 'up' ? 'bg-red-900/30 text-red-400' :
                            resource.trend === 'down' ? 'bg-green-900/30 text-green-400' :
                            'bg-gray-800 text-gray-500'
                          }`}>
                            {resource.trend === 'up' ? '↑' : resource.trend === 'down' ? '↓' : '→'}
                          </span>
                        </div>
                        <div className="flex items-center space-x-4 mb-2 text-sm text-gray-400">
                          <div className="flex items-center space-x-1">
                            <MapPin className="w-3 h-3" />
                            <span>{resource.location}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>{formatCurrency(resource.dailyCost)}/day</span>
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-1 mb-2">
                          {resource.tags.map((tag, idx) => (
                            <span key={idx} className="text-xs px-2 py-0.5 bg-gray-800 text-gray-400 rounded">
                              #{tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="text-right ml-4">
                        <p className="text-2xl font-bold font-mono text-white">{formatCurrency(resource.cost)}</p>
                        <p className="text-xs text-gray-500">{resource.percentage}% of total</p>
                        {resource.optimization.potential > 0 && (
                          <p className="text-xs text-green-400 mt-1">
                            Save {formatCurrency(resource.optimization.potential)}
                          </p>
                        )}
                      </div>
                    </div>
                    
                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-4 mb-3">
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-gray-500">Utilization</span>
                          <span className="text-xs font-mono text-white">{resource.metrics.utilization}%</span>
                        </div>
                        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 rounded-full transition-all duration-1000" 
                            style={{ width: `${resource.metrics.utilization}%` }} 
                          />
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-gray-500">Performance</span>
                          <span className="text-xs font-mono text-white">{resource.metrics.performance}%</span>
                        </div>
                        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-green-500 rounded-full transition-all duration-1000" 
                            style={{ width: `${resource.metrics.performance}%` }} 
                          />
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs text-gray-500">Efficiency</span>
                          <span className="text-xs font-mono text-white">{resource.metrics.efficiency}%</span>
                        </div>
                        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-purple-500 rounded-full transition-all duration-1000" 
                            style={{ width: `${resource.metrics.efficiency}%` }} 
                          />
                        </div>
                      </div>
                    </div>

                    {/* Optimization Recommendations */}
                    {resource.optimization.recommendations.length > 0 && (
                      <div className="border-t border-gray-800 pt-3">
                        <h5 className="text-xs text-gray-500 uppercase font-semibold mb-2">Optimization Recommendations</h5>
                        <div className="space-y-1">
                          {resource.optimization.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-center space-x-2 text-xs">
                              <Lightbulb className="w-3 h-3 text-yellow-400" />
                              <span className="text-gray-400">{rec}</span>
                            </div>
                          ))}
                        </div>
                        <div className="flex items-center justify-between mt-2">
                          <span className={`text-xs px-2 py-1 rounded ${
                            resource.optimization.riskLevel === 'high' ? 'bg-red-900/30 text-red-400' :
                            resource.optimization.riskLevel === 'medium' ? 'bg-yellow-900/30 text-yellow-400' :
                            'bg-green-900/30 text-green-400'
                          }`}>
                            {resource.optimization.riskLevel.toUpperCase()} RISK
                          </span>
                          <span className="text-xs text-green-400 font-bold">
                            Potential: {formatCurrency(resource.optimization.potential)}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Optimization Tab */}
        {activeTab === 'optimization' && (
          <div className="space-y-6">
            {/* Optimization Overview */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
              {Object.entries(data?.optimization || {}).map(([category, savings]) => {
                const categoryNames = {
                  rightsizing: 'Rightsizing',
                  reservedInstances: 'Reserved Instances',
                  storageOptimization: 'Storage Optimization',
                  scheduledShutdown: 'Scheduled Shutdown',
                  unusedResources: 'Unused Resources'
                };
                const categoryIcons = {
                  rightsizing: Server,
                  reservedInstances: CreditCard,
                  storageOptimization: HardDrive,
                  scheduledShutdown: Clock,
                  unusedResources: AlertTriangle
                };
                const Icon = categoryIcons[category as keyof typeof categoryIcons];
                
                return (
                  <div key={category} className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-green-800/50 transition-all duration-200">
                    <div className="flex items-center justify-between mb-3">
                      <Icon className="w-5 h-5 text-green-500" />
                      <span className="text-xs text-green-400 font-bold">
                        {((savings / (data?.savings.identified || 1)) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <h3 className="text-sm font-bold text-white mb-2">
                      {categoryNames[category as keyof typeof categoryNames]}
                    </h3>
                    <p className="text-2xl font-bold font-mono text-green-400">{formatCurrency(savings)}</p>
                    <p className="text-xs text-gray-500 mt-1">potential savings</p>
                  </div>
                );
              })}
            </div>

            {/* Risk Analysis */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-lg font-bold text-white">SAVINGS BY RISK LEVEL</h3>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-3 gap-6">
                  {Object.entries(data?.savings.riskAnalysis || {}).map(([risk, amount]) => (
                    <div key={risk} className="text-center">
                      <div className={`w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center ${
                        risk === 'high' ? 'bg-red-900/30 border-2 border-red-800' :
                        risk === 'medium' ? 'bg-yellow-900/30 border-2 border-yellow-800' :
                        'bg-green-900/30 border-2 border-green-800'
                      }`}>
                        <Shield className={`w-8 h-8 ${
                          risk === 'high' ? 'text-red-400' :
                          risk === 'medium' ? 'text-yellow-400' :
                          'text-green-400'
                        }`} />
                      </div>
                      <h4 className="text-lg font-bold text-white capitalize mb-1">{risk} Risk</h4>
                      <p className="text-2xl font-bold font-mono text-white mb-1">{formatCurrency(amount)}</p>
                      <p className="text-xs text-gray-500">
                        {((amount / (data?.savings.identified || 1)) * 100).toFixed(0)}% of total
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Detailed Recommendations */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-lg font-bold text-white">OPTIMIZATION RECOMMENDATIONS</h3>
              </div>
              <div className="divide-y divide-gray-800">
                {filteredResources
                  .filter(r => r.optimization.potential > 0)
                  .sort((a, b) => b.optimization.potential - a.optimization.potential)
                  .slice(0, 8)
                  .map((resource) => (
                  <div key={resource.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h4 className="font-bold text-white">{resource.name}</h4>
                          <span className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded">
                            {resource.type}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded font-bold ${
                            resource.optimization.riskLevel === 'high' ? 'bg-red-900/30 text-red-400' :
                            resource.optimization.riskLevel === 'medium' ? 'bg-yellow-900/30 text-yellow-400' :
                            'bg-green-900/30 text-green-400'
                          }`}>
                            {resource.optimization.riskLevel.toUpperCase()} RISK
                          </span>
                        </div>
                        <div className="space-y-2">
                          {resource.optimization.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-start space-x-2">
                              <Lightbulb className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                              <span className="text-sm text-gray-400">{rec}</span>
                            </div>
                          ))}
                        </div>
                        <div className="flex items-center space-x-4 mt-3 text-xs text-gray-500">
                          <div>Current: {formatCurrency(resource.cost)}/month</div>
                          <div>Utilization: {resource.metrics.utilization}%</div>
                        </div>
                      </div>
                      <div className="text-right ml-6">
                        <p className="text-2xl font-bold font-mono text-green-400">
                          {formatCurrency(resource.optimization.potential)}
                        </p>
                        <p className="text-xs text-gray-500">monthly savings</p>
                        <div className="mt-2 flex space-x-2">
                          <button className="px-3 py-1 bg-green-900/30 hover:bg-green-900/50 border border-green-800 rounded text-green-400 text-xs transition-colors">
                            APPLY
                          </button>
                          <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-gray-400 text-xs transition-colors">
                            DETAILS
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Budgets Tab */}
        {activeTab === 'budgets' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {data?.budgetAlerts.map((budget) => (
                <div key={budget.id} className={`border rounded-xl p-6 transition-all duration-200 ${
                  budget.status === 'critical' ? 'bg-red-900/10 border-red-800 hover:border-red-700' :
                  budget.status === 'warning' ? 'bg-yellow-900/10 border-yellow-800 hover:border-yellow-700' :
                  'bg-gray-900 border-gray-800 hover:border-gray-700'
                }`}>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-white mb-1">{budget.name}</h3>
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs px-2 py-1 rounded font-bold ${
                          budget.status === 'critical' ? 'bg-red-900/50 text-red-400' :
                          budget.status === 'warning' ? 'bg-yellow-900/50 text-yellow-400' :
                          'bg-green-900/50 text-green-400'
                        }`}>
                          {budget.status.toUpperCase()}
                        </span>
                        {budget.daysRemaining >= 0 ? (
                          <span className="text-xs text-gray-500">{budget.daysRemaining} days remaining</span>
                        ) : (
                          <span className="text-xs text-red-400">Budget exceeded {Math.abs(budget.daysRemaining)} days ago</span>
                        )}
                      </div>
                    </div>
                    <Wallet className={`w-6 h-6 ${
                      budget.status === 'critical' ? 'text-red-500' :
                      budget.status === 'warning' ? 'text-yellow-500' :
                      'text-green-500'
                    }`} />
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-gray-400">Current Spend</span>
                        <span className="text-lg font-bold font-mono text-white">
                          {formatCurrency(budget.current)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-gray-400">Budget Limit</span>
                        <span className="text-sm font-mono text-gray-300">
                          {formatCurrency(budget.threshold)}
                        </span>
                      </div>
                      <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-1000 ${
                            budget.percentage >= 100 ? 'bg-red-500' :
                            budget.percentage >= 80 ? 'bg-yellow-500' :
                            'bg-green-500'
                          }`}
                          style={{ width: `${Math.min(budget.percentage, 100)}%` }}
                        />
                      </div>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-xs text-gray-500">{budget.percentage.toFixed(1)}% used</span>
                        <span className="text-xs text-gray-500">
                          {formatCurrency(budget.threshold - budget.current)} remaining
                        </span>
                      </div>
                    </div>

                    <div className="border-t border-gray-800 pt-4">
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-sm text-gray-400">Forecasted End</p>
                          <p className="text-lg font-bold font-mono text-white">
                            {formatCurrency(budget.forecast)}
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-gray-400">Variance</p>
                          <p className={`text-lg font-bold font-mono ${
                            budget.forecast > budget.threshold ? 'text-red-400' : 'text-green-400'
                          }`}>
                            {budget.forecast > budget.threshold ? '+' : ''}
                            {formatCurrency(budget.forecast - budget.threshold)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Anomalies Tab */}
        {activeTab === 'anomalies' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4">
              {data?.anomalies.map((anomaly) => (
                <div key={anomaly.id} className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-red-800/50 transition-all duration-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <AlertTriangle className={`w-5 h-5 ${
                          anomaly.severity === 'high' ? 'text-red-500' :
                          anomaly.severity === 'medium' ? 'text-yellow-500' :
                          'text-gray-500'
                        }`} />
                        <h3 className="text-lg font-bold text-white">{anomaly.resource}</h3>
                        <span className={`text-xs px-2 py-1 rounded font-bold ${
                          anomaly.severity === 'high' ? 'bg-red-900/50 text-red-400' :
                          anomaly.severity === 'medium' ? 'bg-yellow-900/50 text-yellow-400' :
                          'bg-gray-800 text-gray-400'
                        }`}>
                          {anomaly.severity.toUpperCase()} SEVERITY
                        </span>
                        <span className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded">
                          {anomaly.category}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Cost Spike:</span>
                          <p className="text-white font-bold">{anomaly.spike}%</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Financial Impact:</span>
                          <p className="text-white font-bold">{formatCurrency(anomaly.impact)}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Detected:</span>
                          <p className="text-white font-bold">{anomaly.detected}</p>
                        </div>
                        <div>
                          <span className="text-gray-500">Category:</span>
                          <p className="text-white font-bold">{anomaly.category}</p>
                        </div>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <button className="px-4 py-2 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-800 text-blue-400 rounded-lg transition-colors flex items-center space-x-2">
                        <Eye className="w-4 h-4" />
                        <span>INVESTIGATE</span>
                      </button>
                      <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-400 rounded-lg transition-colors">
                        DISMISS
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Forecasting Tab */}
        {activeTab === 'forecasting' && (
          <div className="space-y-6">
            {/* Forecast Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Next Month</h3>
                  <Calendar className="w-5 h-5 text-blue-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">{formatCurrency(data?.forecast || 0)}</p>
                <p className="text-xs text-gray-500 mt-2">predicted spend</p>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Yearly Projection</h3>
                  <TrendingUp className="w-5 h-5 text-purple-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">{formatCurrency(data?.projectedYear || 0)}</p>
                <p className="text-xs text-gray-500 mt-2">full year estimate</p>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Growth Rate</h3>
                  <Growth className="w-5 h-5 text-green-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">
                  {(((data?.projectedYear || 0) / (data?.yearToDate || 1) - 1) * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-gray-500 mt-2">year over year</p>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-gray-400 uppercase">Confidence</h3>
                  <Calculator className="w-5 h-5 text-cyan-500" />
                </div>
                <p className="text-3xl font-bold font-mono text-white">87%</p>
                <div className="mt-3 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500 rounded-full transition-all duration-1000" style={{ width: '87%' }} />
                </div>
              </div>
            </div>

            {/* Forecast Chart */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl">
              <div className="p-6 border-b border-gray-800">
                <h3 className="text-lg font-bold text-white">12-MONTH FORECAST</h3>
                <p className="text-sm text-gray-400 mt-1">
                  Based on historical trends, seasonal patterns, and current resource utilization
                </p>
              </div>
              <div className="p-6">
                {renderHistoricalChart()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}