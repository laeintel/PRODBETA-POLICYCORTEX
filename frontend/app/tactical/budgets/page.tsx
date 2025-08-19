'use client';

import { useState, useMemo } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { PieChart, TrendingUp, TrendingDown, AlertTriangle, Target, DollarSign, Calendar, Plus, Settings, Download } from 'lucide-react';

export default function Page() {
  const [selectedPeriod, setSelectedPeriod] = useState('monthly');
  const [selectedBudget, setSelectedBudget] = useState('all');

  // Mock budget data
  const budgets = [
    {
      id: '1',
      name: 'Production Environment',
      allocated: 25000,
      spent: 21340,
      forecast: 23800,
      period: 'monthly',
      status: 'warning',
      departments: ['Engineering', 'DevOps'],
      trend: 8.5
    },
    {
      id: '2',
      name: 'Development Environment',
      allocated: 8000,
      spent: 5420,
      forecast: 6200,
      period: 'monthly',
      status: 'healthy',
      departments: ['Engineering'],
      trend: -2.1
    },
    {
      id: '3',
      name: 'Data & Analytics',
      allocated: 15000,
      spent: 14200,
      forecast: 16800,
      period: 'monthly',
      status: 'critical',
      departments: ['Data Science', 'BI'],
      trend: 12.3
    },
    {
      id: '4',
      name: 'Security & Compliance',
      allocated: 6000,
      spent: 4800,
      forecast: 5400,
      period: 'monthly',
      status: 'healthy',
      departments: ['Security'],
      trend: -5.2
    }
  ];

  const totalAllocated = budgets.reduce((sum, b) => sum + b.allocated, 0);
  const totalSpent = budgets.reduce((sum, b) => sum + b.spent, 0);
  const totalForecast = budgets.reduce((sum, b) => sum + b.forecast, 0);
  const utilizationRate = (totalSpent / totalAllocated) * 100;

  const alerts = [
    { type: 'critical', message: 'Data & Analytics budget exceeded 95% threshold', budget: 'Data & Analytics' },
    { type: 'warning', message: 'Production environment trending 8.5% over budget', budget: 'Production Environment' },
    { type: 'info', message: 'Development environment under-utilized by 32%', budget: 'Development Environment' }
  ];

  const forecastData = [
    { month: 'Jan', budgeted: 54000, actual: 45600, forecast: 48200 },
    { month: 'Feb', budgeted: 54000, actual: 49800, forecast: 52100 },
    { month: 'Mar', budgeted: 54000, actual: 52400, forecast: 54900 },
    { month: 'Apr', budgeted: 54000, actual: null, forecast: 56200 }
  ];

  const content = (
    <div className="space-y-8">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <select 
            value={selectedPeriod} 
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="monthly">Monthly View</option>
            <option value="quarterly">Quarterly View</option>
            <option value="yearly">Annual View</option>
          </select>
          <select 
            value={selectedBudget} 
            onChange={(e) => setSelectedBudget(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Budgets</option>
            <option value="production">Production Only</option>
            <option value="development">Development Only</option>
          </select>
        </div>
        <div className="flex items-center space-x-3">
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <Plus className="w-4 h-4" />
            <span>New Budget</span>
          </button>
          <button className="bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 backdrop-blur-md rounded-xl border border-blue-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <DollarSign className="w-8 h-8 text-blue-400" />
            <div className="text-sm text-blue-300">{budgets.length} budgets</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${totalAllocated.toLocaleString()}</p>
          <p className="text-blue-300 text-sm">Total Allocated</p>
        </div>

        <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 backdrop-blur-md rounded-xl border border-green-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <Target className="w-8 h-8 text-green-400" />
            <div className="text-sm text-green-300">{utilizationRate.toFixed(1)}%</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${totalSpent.toLocaleString()}</p>
          <p className="text-green-300 text-sm">Total Spent</p>
        </div>

        <div className="bg-gradient-to-br from-purple-900/50 to-purple-800/30 backdrop-blur-md rounded-xl border border-purple-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="w-8 h-8 text-purple-400" />
            <div className="text-sm text-purple-300">Projected</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${totalForecast.toLocaleString()}</p>
          <p className="text-purple-300 text-sm">Forecast</p>
        </div>

        <div className="bg-gradient-to-br from-red-900/50 to-red-800/30 backdrop-blur-md rounded-xl border border-red-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8 text-red-400" />
            <div className="text-sm text-red-300">{alerts.filter(a => a.type === 'critical').length}</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">{alerts.length}</p>
          <p className="text-red-300 text-sm">Active Alerts</p>
        </div>
      </div>

      {/* Budget Alerts */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <AlertTriangle className="w-5 h-5 text-yellow-500 mr-2" />
          Budget Alerts
        </h3>
        <div className="space-y-3">
          {alerts.map((alert, index) => (
            <div key={index} className={`p-4 rounded-lg border flex items-center justify-between ${
              alert.type === 'critical' ? 'bg-red-900/20 border-red-500/30' :
              alert.type === 'warning' ? 'bg-yellow-900/20 border-yellow-500/30' :
              'bg-blue-900/20 border-blue-500/30'
            }`}>
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full ${
                  alert.type === 'critical' ? 'bg-red-500' :
                  alert.type === 'warning' ? 'bg-yellow-500' :
                  'bg-blue-500'
                }`} />
                <div>
                  <p className="text-white font-medium">{alert.message}</p>
                  <p className="text-gray-400 text-sm">Budget: {alert.budget}</p>
                </div>
              </div>
              <button className="text-blue-400 hover:text-blue-300 text-sm">
                View Details
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Budget Overview Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800">
          <h3 className="text-xl font-bold text-white">Budget Overview</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Budget Name</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Allocated</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Spent</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Utilization</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Forecast</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Trend</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {budgets.map((budget) => {
                const utilization = (budget.spent / budget.allocated) * 100;
                return (
                  <tr key={budget.id} className="hover:bg-gray-800/30 transition-colors">
                    <td className="px-6 py-4">
                      <div>
                        <p className="text-white font-medium">{budget.name}</p>
                        <p className="text-gray-400 text-sm">{budget.departments.join(', ')}</p>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-white font-semibold">${budget.allocated.toLocaleString()}</td>
                    <td className="px-6 py-4 text-gray-300">${budget.spent.toLocaleString()}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-full rounded-full transition-all duration-500 ${
                              utilization > 95 ? 'bg-red-500' : 
                              utilization > 80 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${Math.min(utilization, 100)}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-300">{utilization.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-300">${budget.forecast.toLocaleString()}</td>
                    <td className="px-6 py-4">
                      <div className={`flex items-center space-x-1 ${
                        budget.trend > 0 ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {budget.trend > 0 ? 
                          <TrendingUp className="w-4 h-4" /> : 
                          <TrendingDown className="w-4 h-4" />
                        }
                        <span className="text-sm">{Math.abs(budget.trend).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        budget.status === 'critical' ? 'bg-red-900/50 text-red-300 border border-red-500/30' :
                        budget.status === 'warning' ? 'bg-yellow-900/50 text-yellow-300 border border-yellow-500/30' :
                        'bg-green-900/50 text-green-300 border border-green-500/30'
                      }`}>
                        {budget.status}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <button className="text-blue-400 hover:text-blue-300 text-sm">
                          Edit
                        </button>
                        <button className="text-gray-400 hover:text-gray-300 text-sm">
                          <Settings className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Budget Forecast Chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Budget vs Actual Trend</h3>
        <div className="space-y-4">
          {forecastData.map((data, index) => (
            <div key={index} className="flex items-center space-x-4">
              <div className="w-12 text-sm text-gray-300 font-medium">{data.month}</div>
              <div className="flex-1">
                <div className="relative">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="text-xs text-gray-400">Budgeted: ${data.budgeted.toLocaleString()}</span>
                    {data.actual && <span className="text-xs text-gray-400">Actual: ${data.actual.toLocaleString()}</span>}
                    <span className="text-xs text-gray-400">Forecast: ${data.forecast.toLocaleString()}</span>
                  </div>
                  <div className="bg-gray-700 rounded-full h-3 relative overflow-hidden">
                    <div className="bg-blue-600 h-full rounded-full" style={{width: '100%'}} />
                    {data.actual && (
                      <div 
                        className={`absolute top-0 left-0 h-full rounded-full ${
                          data.actual > data.budgeted ? 'bg-red-500' : 'bg-green-500'
                        }`}
                        style={{width: `${Math.min((data.actual / data.budgeted) * 100, 100)}%`}}
                      />
                    )}
                    <div 
                      className="absolute top-0 left-0 h-full bg-yellow-500/70 rounded-full"
                      style={{width: `${Math.min((data.forecast / data.budgeted) * 100, 100)}%`}}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-center space-x-6 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-600 rounded-full" />
            <span className="text-gray-400">Budgeted</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span className="text-gray-400">Actual</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500/70 rounded-full" />
            <span className="text-gray-400">Forecast</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Budget Tracking" 
      subtitle="Advanced Budget Management & Forecasting" 
      icon={PieChart}
    >
      {content}
    </TacticalPageTemplate>
  );
}