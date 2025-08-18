'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '../../lib/api-client';
import NextLink from 'next/link';
import { 
  Bell, RefreshCw, Shield, Globe, ArrowUpRight, TrendingUp, 
  CheckCircle, TrendingDown, Server, DollarSign, Cpu, Archive,
  Upload, Zap, Power
} from 'lucide-react';

export default function TacticalDashboard() {
  const router = useRouter();
  const [systemStatus, setSystemStatus] = useState('INITIALIZING');
  const [refreshing, setRefreshing] = useState(false);
  
  // Real-time data states
  const [metrics, setMetrics] = useState<any>({});
  const [threats, setThreats] = useState<any[]>([]);
  const [compliance, setCompliance] = useState<any>({});
  const [resources, setResources] = useState<any[]>([]);
  const [costData, setCostData] = useState<any>({});
  const [predictions, setPredictions] = useState<any[]>([]);
  const [correlations, setCorrelations] = useState<any[]>([]);
  const [activityFeed, setActivityFeed] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);

  // Fetch all data
  const fetchDashboardData = async () => {
    setRefreshing(true);
    try {
      const [
        metricsRes,
        threatsRes, 
        complianceRes,
        resourcesRes,
        costRes,
        predictionsRes,
        correlationsRes
      ] = await Promise.all([
        api.getUnifiedMetrics(),
        api.getSecurityThreats(),
        api.getComplianceStatus(),
        api.getResources(),
        api.getCostAnalysis(),
        api.getAIPredictions(),
        api.getCorrelations()
      ]);

      if (metricsRes.data) setMetrics(metricsRes.data);
      if (threatsRes.data) setThreats(threatsRes.data.activeThreats || []);
      if (complianceRes.data) setCompliance(complianceRes.data);
      if (resourcesRes.data) setResources(resourcesRes.data.resources || []);
      if (costRes.data) setCostData(costRes.data);
      if (predictionsRes.data) setPredictions(predictionsRes.data.predictions || []);
      if (correlationsRes.data) setCorrelations(correlationsRes.data.correlations || []);
      
      // Simulate activity feed
      setActivityFeed([
        { id: 1, type: 'policy', message: 'Policy enforcement completed', time: '2 min ago', severity: 'info' },
        { id: 2, type: 'security', message: 'Threat detected and mitigated', time: '5 min ago', severity: 'warning' },
        { id: 3, type: 'cost', message: 'Cost optimization saved $2,847', time: '12 min ago', severity: 'success' },
        { id: 4, type: 'compliance', message: 'Compliance scan initiated', time: '15 min ago', severity: 'info' },
        { id: 5, type: 'resource', message: 'Auto-scaling triggered for VM cluster', time: '22 min ago', severity: 'info' }
      ]);

      setAlerts([
        { id: 1, title: 'High CPU Usage', resource: 'VM-PROD-01', severity: 'high', time: '1 min ago' },
        { id: 2, title: 'SSL Certificate Expiring', resource: 'webapp.com', severity: 'medium', time: '1 hour ago' },
        { id: 3, title: 'Backup Failed', resource: 'DB-BACKUP-01', severity: 'critical', time: '2 hours ago' }
      ]);

      setSystemStatus('OPERATIONAL');
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setSystemStatus('DEGRADED');
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const executeAction = async (actionType: string, resourceId: string = 'global') => {
    try {
      const resp = await api.createAction(resourceId, actionType, {});
      if (resp.data?.action_id) {
        console.log(`Action ${actionType} initiated`);
      }
    } catch (error) {
      console.error('Action failed:', error);
    }
  };

  return (
    <>
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold">Tactical Operations Dashboard</h1>
            <div className={`px-2 py-1 rounded text-xs font-medium ${
              systemStatus === 'OPERATIONAL' ? 'bg-green-900/30 text-green-500' :
              systemStatus === 'DEGRADED' ? 'bg-yellow-900/30 text-yellow-500' :
              'bg-gray-900/30 text-gray-500'
            }`}>
              {systemStatus}
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={fetchDashboardData}
              className={`p-2 hover:bg-gray-800 rounded ${refreshing ? 'animate-spin' : ''}`}
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button className="p-2 hover:bg-gray-800 rounded relative">
              <Bell className="w-4 h-4" />
              {alerts.length > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                  {alerts.length}
                </span>
              )}
            </button>
            <button className="flex items-center space-x-2 px-3 py-1 bg-gray-800 rounded">
              <Globe className="w-3 h-3" />
              <span className="text-xs">EAST US</span>
            </button>
          </div>
        </div>
      </header>

      {/* Dashboard Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Key Metrics Row */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <MetricCard
            title="Resources"
            value={metrics.resources || 0}
            trend={12}
            icon={Server}
            color="blue"
          />
          <MetricCard
            title="Compliance"
            value={`${metrics.compliance || 0}%`}
            trend={2.3}
            icon={CheckCircle}
            color="green"
          />
          <MetricCard
            title="Active Threats"
            value={metrics.threats || 0}
            trend={-15}
            icon={Shield}
            color="red"
          />
          <MetricCard
            title="Monthly Cost"
            value={`$${(metrics.cost || 0).toLocaleString()}`}
            trend={-8}
            icon={DollarSign}
            color="yellow"
          />
          <MetricCard
            title="AI Predictions"
            value={metrics.predictions || 0}
            trend={47}
            icon={Cpu}
            color="purple"
          />
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Security Threats Monitor */}
          <div className="col-span-4 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold">Security Threats</h3>
              <NextLink href="/tactical/security" className="text-xs text-gray-400 hover:text-gray-200">
                View All
              </NextLink>
            </div>
            <div className="space-y-2">
              {threats.slice(0, 5).map((threat) => (
                <div key={threat.id} className="p-3 bg-gray-800 rounded">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm font-medium">{threat.type}</p>
                      <p className="text-xs text-gray-400">Source: {threat.source}</p>
                      <p className="text-xs text-gray-500">{threat.detected}</p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${
                      threat.severity === 'high' ? 'bg-red-900/30 text-red-500' :
                      threat.severity === 'medium' ? 'bg-yellow-900/30 text-yellow-500' :
                      'bg-blue-900/30 text-blue-500'
                    }`}>
                      {threat.severity}
                    </span>
                  </div>
                  <button
                    onClick={() => executeAction('mitigate_threat', threat.id)}
                    className="mt-2 w-full px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded"
                  >
                    Mitigate
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Compliance Status */}
          <div className="col-span-4 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold">Compliance Status</h3>
              <button 
                onClick={() => executeAction('compliance_scan')}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded"
              >
                Scan Now
              </button>
            </div>
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400">Overall Score</span>
                <span className="text-2xl font-bold text-green-500">
                  {compliance.overallScore || 0}%
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${compliance.overallScore || 0}%` }}
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="p-2 bg-gray-800 rounded">
                <p className="text-xs text-gray-400">Compliant</p>
                <p className="text-lg font-bold text-green-500">{compliance.compliant || 0}</p>
              </div>
              <div className="p-2 bg-gray-800 rounded">
                <p className="text-xs text-gray-400">Non-Compliant</p>
                <p className="text-lg font-bold text-red-500">{compliance.nonCompliant || 0}</p>
              </div>
              <div className="p-2 bg-gray-800 rounded">
                <p className="text-xs text-gray-400">Total Resources</p>
                <p className="text-lg font-bold">{compliance.totalResources || 0}</p>
              </div>
              <div className="p-2 bg-gray-800 rounded">
                <p className="text-xs text-gray-400">Policies</p>
                <p className="text-lg font-bold">{metrics.policies || 0}</p>
              </div>
            </div>
          </div>

          {/* Cost Analysis */}
          <div className="col-span-4 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold">Cost Analysis</h3>
              <NextLink href="/tactical/cost" className="text-xs text-gray-400 hover:text-gray-200">
                Details
              </NextLink>
            </div>
            <div className="mb-4">
              <p className="text-xs text-gray-400 mb-1">Current Month</p>
              <p className="text-2xl font-bold">${(costData.currentMonth || 0).toLocaleString()}</p>
              <div className="flex items-center mt-1">
                {costData.trend < 0 ? (
                  <>
                    <TrendingDown className="w-3 h-3 text-green-500 mr-1" />
                    <span className="text-xs text-green-500">{Math.abs(costData.trend)}% from last month</span>
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-3 h-3 text-red-500 mr-1" />
                    <span className="text-xs text-red-500">{costData.trend}% from last month</span>
                  </>
                )}
              </div>
            </div>
            <div className="space-y-2">
              {Object.entries(costData.breakdown || {}).slice(0, 4).map(([category, amount]) => (
                <div key={category} className="flex items-center justify-between">
                  <span className="text-xs text-gray-400 capitalize">{category}</span>
                  <span className="text-xs font-medium">${(amount as number).toLocaleString()}</span>
                </div>
              ))}
            </div>
            <button
              onClick={() => executeAction('optimize_costs')}
              className="mt-3 w-full px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white text-xs rounded"
            >
              Optimize Costs
            </button>
          </div>

          {/* AI Predictions Panel */}
          <div className="col-span-6 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold">AI Predictions</h3>
              <button 
                onClick={() => executeAction('run_predictions')}
                className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs rounded"
              >
                Run Analysis
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {predictions.map((pred) => (
                <div key={pred.id} className="p-3 bg-gray-800 rounded">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium">{pred.type.replace(/_/g, ' ').toUpperCase()}</span>
                    <span className={`text-xs font-bold ${
                      pred.probability > 75 ? 'text-red-500' :
                      pred.probability > 50 ? 'text-yellow-500' :
                      'text-green-500'
                    }`}>
                      {pred.probability}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div 
                      className={`h-1.5 rounded-full ${
                        pred.probability > 75 ? 'bg-red-500' :
                        pred.probability > 50 ? 'bg-yellow-500' :
                        'bg-green-500'
                      }`}
                      style={{ width: `${pred.probability}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-400 mt-1">Within {pred.timeframe}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Resource Utilization */}
          <div className="col-span-3 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold mb-4">Resource Utilization</h3>
            <div className="space-y-3">
              <UtilizationBar label="CPU" value={73} color="blue" />
              <UtilizationBar label="Memory" value={61} color="green" />
              <UtilizationBar label="Storage" value={84} color="yellow" />
              <UtilizationBar label="Network" value={42} color="purple" />
              <UtilizationBar label="Database" value={91} color="red" />
            </div>
          </div>

          {/* Activity Feed */}
          <div className="col-span-3 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold mb-4">Activity Feed</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {activityFeed.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-2">
                  <div className={`w-2 h-2 rounded-full mt-1.5 ${
                    activity.severity === 'success' ? 'bg-green-500' :
                    activity.severity === 'warning' ? 'bg-yellow-500' :
                    activity.severity === 'error' ? 'bg-red-500' :
                    'bg-blue-500'
                  }`} />
                  <div className="flex-1">
                    <p className="text-xs">{activity.message}</p>
                    <p className="text-xs text-gray-500">{activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Correlation Analysis */}
          <div className="col-span-6 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold mb-4">Cross-Domain Correlations</h3>
            <div className="grid grid-cols-3 gap-3">
              {correlations.map((corr) => (
                <div key={corr.id} className="p-3 bg-gray-800 rounded">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs">{corr.source}</span>
                    <ArrowUpRight className="w-3 h-3 text-gray-500" />
                    <span className="text-xs">{corr.target}</span>
                  </div>
                  <div className="text-center">
                    <span className={`text-lg font-bold ${
                      corr.strength > 80 ? 'text-green-500' :
                      corr.strength > 60 ? 'text-yellow-500' :
                      'text-red-500'
                    }`}>
                      {corr.strength}%
                    </span>
                    <p className="text-xs text-gray-400">{corr.type} correlation</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="col-span-6 bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold mb-4">Quick Actions</h3>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => executeAction('security_scan')}
                className="p-3 bg-gray-800 hover:bg-gray-700 rounded flex flex-col items-center space-y-1"
              >
                <Shield className="w-5 h-5 text-blue-500" />
                <span className="text-xs">Security Scan</span>
              </button>
              <button
                onClick={() => executeAction('backup_now')}
                className="p-3 bg-gray-800 hover:bg-gray-700 rounded flex flex-col items-center space-y-1"
              >
                <Archive className="w-5 h-5 text-green-500" />
                <span className="text-xs">Backup Now</span>
              </button>
              <button
                onClick={() => executeAction('deploy')}
                className="p-3 bg-gray-800 hover:bg-gray-700 rounded flex flex-col items-center space-y-1"
              >
                <Upload className="w-5 h-5 text-cyan-500" />
                <span className="text-xs">Deploy</span>
              </button>
              <button
                onClick={() => executeAction('optimize')}
                className="p-3 bg-gray-800 hover:bg-gray-700 rounded flex flex-col items-center space-y-1"
              >
                <Zap className="w-5 h-5 text-yellow-500" />
                <span className="text-xs">Optimize</span>
              </button>
              <button
                onClick={() => executeAction('train_models')}
                className="p-3 bg-gray-800 hover:bg-gray-700 rounded flex flex-col items-center space-y-1"
              >
                <Cpu className="w-5 h-5 text-purple-500" />
                <span className="text-xs">Train AI</span>
              </button>
              <button
                onClick={() => executeAction('emergency_lockdown')}
                className="p-3 bg-red-900/20 hover:bg-red-900/30 rounded flex flex-col items-center space-y-1 border border-red-800"
              >
                <Power className="w-5 h-5 text-red-500" />
                <span className="text-xs text-red-500">Emergency</span>
              </button>
            </div>
          </div>

          {/* Critical Alerts */}
          <div className="col-span-12 bg-red-900/10 border border-red-900/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-red-500">Critical Alerts</h3>
              <NextLink href="/tactical/alerts" className="text-xs text-gray-400 hover:text-gray-200">
                Manage Alerts
              </NextLink>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="p-3 bg-gray-900 rounded border border-gray-800">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm font-medium">{alert.title}</p>
                      <p className="text-xs text-gray-400">{alert.resource}</p>
                      <p className="text-xs text-gray-500">{alert.time}</p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded ${
                      alert.severity === 'critical' ? 'bg-red-900/30 text-red-500' :
                      alert.severity === 'high' ? 'bg-orange-900/30 text-orange-500' :
                      'bg-yellow-900/30 text-yellow-500'
                    }`}>
                      {alert.severity}
                    </span>
                  </div>
                  <div className="mt-2 flex space-x-2">
                    <button className="flex-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 text-xs rounded">
                      Investigate
                    </button>
                    <button className="flex-1 px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded">
                      Resolve
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// Metric Card Component
function MetricCard({ title, value, trend, icon: Icon, color }: any) {
  const colorClasses = {
    blue: 'text-blue-500 bg-blue-900/20',
    green: 'text-green-500 bg-green-900/20',
    red: 'text-red-500 bg-red-900/20',
    yellow: 'text-yellow-500 bg-yellow-900/20',
    purple: 'text-purple-500 bg-purple-900/20'
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-gray-400">{title}</p>
        <div className={`p-1.5 rounded ${colorClasses[color]}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <p className="text-2xl font-bold mb-1">{value}</p>
      <div className="flex items-center">
        {trend > 0 ? (
          <>
            <TrendingUp className="w-3 h-3 text-green-500 mr-1" />
            <span className="text-xs text-green-500">+{trend}%</span>
          </>
        ) : (
          <>
            <TrendingDown className="w-3 h-3 text-red-500 mr-1" />
            <span className="text-xs text-red-500">{trend}%</span>
          </>
        )}
      </div>
    </div>
  );
}

// Utilization Bar Component
function UtilizationBar({ label, value, color }: any) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    purple: 'bg-purple-500',
    red: 'bg-red-500'
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-400">{label}</span>
        <span className="text-xs font-medium">{value}%</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5">
        <div 
          className={`h-1.5 rounded-full ${colorClasses[color]}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}