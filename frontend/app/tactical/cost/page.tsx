'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import AuthGuard from '../../../components/AuthGuard';
import { api } from '../../../lib/api-client';
import toast from 'react-hot-toast';
import { DollarSign, TrendingUp, TrendingDown, AlertTriangle, PieChart, BarChart3, Calendar, Download } from 'lucide-react';

interface CostData {
  currentMonth: number;
  previousMonth: number;
  trend: number;
  forecast: number;
  breakdown: {
    compute: number;
    storage: number;
    networking: number;
    database: number;
    other: number;
  };
  topSpenders: Array<{
    id: string;
    name: string;
    cost: number;
    trend: 'up' | 'down' | 'stable';
    percentage: number;
  }>;
  anomalies: Array<{
    id: string;
    resource: string;
    spike: number;
    detected: string;
    severity: 'low' | 'medium' | 'high';
  }>;
  savings: {
    identified: number;
    implemented: number;
    recommendations: number;
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
  const [timeRange, setTimeRange] = useState('30d');
  const [view, setView] = useState<'overview' | 'breakdown' | 'optimization'>('overview');

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
    breakdown: {
      compute: 45678,
      storage: 23456,
      networking: 12345,
      database: 34567,
      other: 11393
    },
    topSpenders: [
      { id: 'r1', name: 'Production VMs', cost: 34567, trend: 'up', percentage: 27 },
      { id: 'r2', name: 'SQL Databases', cost: 28934, trend: 'down', percentage: 23 },
      { id: 'r3', name: 'Storage Accounts', cost: 19234, trend: 'stable', percentage: 15 },
      { id: 'r4', name: 'App Services', cost: 15678, trend: 'up', percentage: 12 },
      { id: 'r5', name: 'Load Balancers', cost: 12345, trend: 'down', percentage: 10 }
    ],
    anomalies: [
      { id: 'a1', resource: 'VM-PROD-01', spike: 234, detected: '2 hours ago', severity: 'high' },
      { id: 'a2', resource: 'Storage-Archive', spike: 156, detected: '1 day ago', severity: 'medium' },
      { id: 'a3', resource: 'CDN-Global', spike: 89, detected: '3 days ago', severity: 'low' }
    ],
    savings: {
      identified: 45000,
      implemented: 28000,
      recommendations: 12
    }
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-green-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-400">LOADING COST ANALYTICS...</p>
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
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/tactical" className="text-gray-400 hover:text-gray-200">
                ← BACK
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">COST ANALYTICS CENTER</h1>
              <div className={`px-3 py-1 rounded text-xs font-bold ${
                data?.trend && data.trend < 0 
                  ? 'bg-green-900/30 text-green-500' 
                  : 'bg-red-900/30 text-red-500'
              }`}>
                {data?.trend && data.trend < 0 ? '↓' : '↑'} {Math.abs(data?.trend || 0)}%
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <select 
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="px-4 py-2 bg-gray-800 border border-gray-700 rounded text-sm"
              >
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="90d">Last 90 Days</option>
                <option value="1y">Last Year</option>
              </select>
              <button onClick={() => triggerAction('optimize_costs')} className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded transition-colors">
                OPTIMIZE COSTS
              </button>
              <button onClick={() => triggerAction('export_cost_report')} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors flex items-center gap-2">
                <Download className="w-4 h-4" />
                EXPORT REPORT
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Current Month</p>
            <p className="text-3xl font-bold font-mono">{formatCurrency(data?.currentMonth || 0)}</p>
            <p className={`text-xs mt-1 ${data?.trend && data.trend < 0 ? 'text-green-500' : 'text-red-500'}`}>
              {data?.trend && data.trend < 0 ? '↓' : '↑'} {formatCurrency(Math.abs((data?.currentMonth || 0) - (data?.previousMonth || 0)))}
            </p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Previous Month</p>
            <p className="text-3xl font-bold font-mono">{formatCurrency(data?.previousMonth || 0)}</p>
            <p className="text-xs text-gray-500 mt-1">baseline</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Forecast</p>
            <p className="text-3xl font-bold font-mono text-yellow-500">{formatCurrency(data?.forecast || 0)}</p>
            <p className="text-xs text-gray-500 mt-1">next 30 days</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Savings Identified</p>
            <p className="text-3xl font-bold font-mono text-green-500">{formatCurrency(data?.savings.identified || 0)}</p>
            <p className="text-xs text-gray-500 mt-1">{data?.savings.recommendations} recommendations</p>
          </div>
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase mb-1">Anomalies</p>
            <p className="text-3xl font-bold font-mono text-orange-500">{data?.anomalies.length}</p>
            <p className="text-xs text-gray-500 mt-1">detected</p>
          </div>
        </div>

        {/* View Tabs */}
        <div className="flex space-x-1 mb-6 bg-gray-900 rounded-lg p-1 inline-flex">
          <button
            onClick={() => setView('overview')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              view === 'overview' 
                ? 'bg-gray-800 text-white' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            OVERVIEW
          </button>
          <button
            onClick={() => setView('breakdown')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              view === 'breakdown' 
                ? 'bg-gray-800 text-white' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            BREAKDOWN
          </button>
          <button
            onClick={() => setView('optimization')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              view === 'optimization' 
                ? 'bg-gray-800 text-white' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            OPTIMIZATION
          </button>
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* Cost Breakdown */}
          <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">SERVICE BREAKDOWN</h3>
            </div>
            <div className="p-6">
              {Object.entries(data?.breakdown || {}).map(([service, cost]) => {
                const percentage = ((cost / (data?.currentMonth || 1)) * 100).toFixed(1);
                return (
                  <div key={service} className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm capitalize">{service}</span>
                      <span className="text-sm font-mono">{formatCurrency(cost)}</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-500 mt-1">{percentage}%</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Top Spenders */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg">
            <div className="p-4 border-b border-gray-800">
              <h3 className="text-sm font-bold text-gray-400 uppercase">TOP SPENDERS</h3>
            </div>
            <div className="divide-y divide-gray-800">
              {data?.topSpenders.map((spender) => (
                <div key={spender.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium text-sm">{spender.name}</h4>
                      <p className="text-2xl font-bold font-mono mt-1">{formatCurrency(spender.cost)}</p>
                    </div>
                    <div className={`text-xs px-2 py-1 rounded ${
                      spender.trend === 'up' ? 'bg-red-900/30 text-red-500' :
                      spender.trend === 'down' ? 'bg-green-900/30 text-green-500' :
                      'bg-gray-800 text-gray-500'
                    }`}>
                      {spender.trend === 'up' ? '↑' : spender.trend === 'down' ? '↓' : '→'}
                    </div>
                  </div>
                  <div className="mt-2">
                    <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-600 rounded-full"
                        style={{ width: `${spender.percentage}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{spender.percentage}% of total</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Cost Anomalies */}
        <div className="mt-6 bg-gray-900 border border-gray-800 rounded-lg">
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-400 uppercase">COST ANOMALIES</h3>
            <AlertTriangle className="w-4 h-4 text-yellow-500" />
          </div>
          <div className="divide-y divide-gray-800">
            {data?.anomalies.map((anomaly) => (
              <div key={anomaly.id} className="p-4 hover:bg-gray-800/50 transition-colors">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">{anomaly.resource}</h4>
                    <p className="text-sm text-gray-500 mt-1">
                      {anomaly.spike}% spike detected {anomaly.detected}
                    </p>
                  </div>
                  <div className={`px-3 py-1 rounded text-xs font-bold ${
                    anomaly.severity === 'high' ? 'bg-red-900/30 text-red-500' :
                    anomaly.severity === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                    'bg-gray-800 text-gray-500'
                  }`}>
                    {anomaly.severity.toUpperCase()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}