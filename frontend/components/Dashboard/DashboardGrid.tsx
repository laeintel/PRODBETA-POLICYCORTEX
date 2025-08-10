'use client';

import React, { useEffect, useState } from 'react';
import { KPITile } from './KPITile';
import { ActionDrawer } from '../ActionDrawer/ActionDrawer';
import { useGovernanceStore } from '@/store/governanceStore';
import { useRealtimeUpdates } from '@/hooks/useRealtimeUpdates';
import MockDataIndicator, { useMockDataStatus } from '../MockDataIndicator';
import { 
  Shield, 
  DollarSign, 
  Users, 
  Activity, 
  AlertTriangle,
  TrendingUp,
  Lock,
  Cloud,
  Cpu,
  Database
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface DashboardMetrics {
  policies: {
    total: number;
    active: number;
    violations: number;
    compliance_rate: number;
    trend: number;
  };
  costs: {
    current_spend: number;
    predicted_spend: number;
    savings_identified: number;
    optimization_rate: number;
    trend: number;
  };
  security: {
    risk_score: number;
    active_threats: number;
    critical_paths: number;
    mitigations_available: number;
    trend: number;
  };
  resources: {
    total: number;
    optimized: number;
    idle: number;
    overprovisioned: number;
    utilization_rate: number;
  };
  compliance: {
    frameworks: number;
    overall_score: number;
    findings: number;
    evidence_packs: number;
    next_assessment_days: number;
  };
  ai: {
    predictions_made: number;
    automations_executed: number;
    accuracy: number;
    learning_progress: number;
  };
}

export function DashboardGrid() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedAction, setSelectedAction] = useState<string | null>(null);
  const [isActionDrawerOpen, setIsActionDrawerOpen] = useState(false);
  const [proactiveActions, setProactiveActions] = useState<any[]>([]);
  const [isUsingFallback, setIsUsingFallback] = useState(false);
  
  // Check if we're using mock data
  const { isMockData } = useMockDataStatus();
  
  // Store is available but not using it directly in this component
  // const store = useGovernanceStore();
  
  // Real-time updates via SSE
  const { isConnected } = useRealtimeUpdates(false); // Disabled for now, using polling instead

  useEffect(() => {
    fetchDashboardMetrics();
    const interval = setInterval(fetchDashboardMetrics, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  // Removed realtime data effect since we're using polling for now

  const fetchDashboardMetrics = async () => {
    try {
      const response = await fetch('/api/v1/metrics');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setMetrics(data);
      setIsUsingFallback(false);
    } catch (error) {
      console.error('Failed to fetch dashboard metrics:', error);
      // Use fallback data if API fails
      setIsUsingFallback(true);
      setMetrics(getFallbackMetrics());
    } finally {
      setLoading(false);
    }
  };
  
  const getFallbackMetrics = (): DashboardMetrics => ({
    policies: {
      total: 45,
      active: 38,
      violations: 7,
      compliance_rate: 84.4,
      trend: 2.3
    },
    costs: {
      current_spend: 127500,
      predicted_spend: 135000,
      savings_identified: 18750,
      optimization_rate: 14.7,
      trend: -3.2
    },
    security: {
      risk_score: 72,
      active_threats: 3,
      critical_paths: 2,
      mitigations_available: 8,
      trend: -5.1
    },
    resources: {
      total: 312,
      optimized: 198,
      idle: 42,
      overprovisioned: 28,
      utilization_rate: 63.5
    },
    compliance: {
      frameworks: 4,
      overall_score: 87,
      findings: 12,
      evidence_packs: 156,
      next_assessment_days: 14
    },
    ai: {
      predictions_made: 2847,
      automations_executed: 412,
      accuracy: 94.3,
      learning_progress: 78
    }
  });

  const handleActionClick = (actionId: string) => {
    setSelectedAction(actionId)
    setIsActionDrawerOpen(true)
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Mock Data Indicator */}
      {(isMockData || isUsingFallback) && (
        <MockDataIndicator 
          type="banner" 
          dataSource={isUsingFallback ? "Fallback Data (API Unavailable)" : "Mock Data"}
          className="mb-4"
        />
      )}
      
      {/* Real-time connection indicator */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Governance Dashboard
        </h1>
        <div className="flex items-center gap-4">
          {!isMockData && !isUsingFallback && (
            <MockDataIndicator type="badge" className="" />
          )}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-400'}`} />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {isConnected ? 'Real-time updates active' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>

      {/* Proactive Actions Alert */}
      {proactiveActions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div>
                <h3 className="font-medium text-blue-900 dark:text-blue-100">
                  {proactiveActions.length} Proactive Actions Available
                </h3>
                <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                  AI has identified opportunities for optimization and risk reduction
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsActionDrawerOpen(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Review Actions
            </button>
          </div>
        </motion.div>
      )}

      {/* Main KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {/* Policy Compliance */}
        <KPITile
          title="Policy Compliance"
          value={metrics ? formatPercentage(metrics.policies.compliance_rate ?? 0) : '--'}
          subtitle={`${metrics?.policies.violations || 0} violations detected`}
          change={metrics?.policies.trend}
          changeLabel="vs last month"
          trend={(metrics?.policies.trend ?? 0) > 0 ? 'up' : 'down'}
          status={(metrics?.policies.compliance_rate ?? 0) >= 99 ? 'success' : 
                  (metrics?.policies.compliance_rate ?? 0) >= 95 ? 'warning' : 'error'}
          deepLink="/policies"
          icon={<Shield className="w-5 h-5 text-blue-600" />}
          loading={loading}
          sparklineData={[98.2, 98.5, 99.1, 99.3, 99.8]}
        />

        {/* Cost Optimization */}
        <KPITile
          title="Monthly Spend"
          value={metrics ? formatCurrency(metrics.costs.current_spend ?? 0) : '--'}
          subtitle={`${formatCurrency(metrics?.costs.savings_identified || 0)} savings identified`}
          change={metrics?.costs.trend}
          changeLabel="vs last month"
          trend={(metrics?.costs.trend ?? 0) < 0 ? 'down' : 'up'}
          status={(metrics?.costs.optimization_rate ?? 0) >= 80 ? 'success' : 'warning'}
          deepLink="/costs"
          icon={<DollarSign className="w-5 h-5 text-green-600" />}
          loading={loading}
          sparklineData={[145000, 142000, 138000, 135000, 132000]}
        />

        {/* Security Risk Score */}
        <KPITile
          title="Security Risk Score"
          value={metrics ? (metrics.security.risk_score ?? 0).toFixed(1) : '--'}
          subtitle={`${metrics?.security.active_threats || 0} active threats`}
          change={metrics?.security.trend}
          changeLabel="vs last week"
          trend={(metrics?.security.trend ?? 0) < 0 ? 'down' : 'up'}
          status={(metrics?.security.risk_score ?? 100) <= 30 ? 'success' : 
                  (metrics?.security.risk_score ?? 100) <= 60 ? 'warning' : 'error'}
          deepLink="/security"
          icon={<Lock className="w-5 h-5 text-red-600" />}
          loading={loading}
          sparklineData={[45, 42, 38, 35, 32]}
        />

        {/* Resource Utilization */}
        <KPITile
          title="Resource Utilization"
          value={metrics ? formatPercentage(metrics.resources.utilization_rate ?? 0) : '--'}
          subtitle={`${metrics?.resources.idle || 0} idle resources`}
          change={5.2}
          changeLabel="efficiency gain"
          trend="up"
          status={(metrics?.resources.utilization_rate ?? 0) >= 70 ? 'success' : 'warning'}
          deepLink="/resources"
          icon={<Cpu className="w-5 h-5 text-purple-600" />}
          loading={loading}
          sparklineData={[65, 68, 70, 72, 75]}
        />

        {/* Compliance Score */}
        <KPITile
          title="Compliance Score"
          value={metrics ? formatPercentage(metrics.compliance.overall_score ?? 0) : '--'}
          subtitle={`${metrics?.compliance.frameworks || 0} frameworks tracked`}
          change={2.1}
          changeLabel="improvement"
          trend="up"
          status={(metrics?.compliance.overall_score ?? 0) >= 95 ? 'success' : 
                  (metrics?.compliance.overall_score ?? 0) >= 80 ? 'warning' : 'error'}
          deepLink="/compliance"
          icon={<Activity className="w-5 h-5 text-indigo-600" />}
          loading={loading}
        />

        {/* AI Predictions */}
        <KPITile
          title="AI Predictions"
          value={metrics ? formatNumber(metrics.ai?.predictions_made ?? 0) : '--'}
          subtitle={`${formatPercentage(metrics?.ai.accuracy || 0)} accuracy`}
          change={metrics?.ai.learning_progress}
          changeLabel="learning progress"
          trend="up"
          status="success"
          deepLink="/ai-insights"
          icon={<Cpu className="w-5 h-5 text-cyan-600" />}
          loading={loading}
        />

        {/* Automations */}
        <KPITile
          title="Automations"
          value={metrics ? formatNumber(metrics.ai?.automations_executed ?? 0) : '--'}
          subtitle="This month"
          change={18.5}
          changeLabel="vs last month"
          trend="up"
          status="success"
          deepLink="/automations"
          icon={<TrendingUp className="w-5 h-5 text-orange-600" />}
          loading={loading}
        />

        {/* Data Governance */}
        <KPITile
          title="Evidence Packs"
          value={metrics?.compliance.evidence_packs || '--'}
          subtitle={`Next assessment in ${metrics?.compliance.next_assessment_days || '--'} days`}
          trend="neutral"
          status="neutral"
          deepLink="/compliance/evidence"
          icon={<Database className="w-5 h-5 text-teal-600" />}
          loading={loading}
        />
      </div>

      {/* Secondary Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-6">
        {/* Quick Stats */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            Quick Stats
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Active Policies</span>
              <span className="font-medium">{metrics?.policies.active || '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Resources</span>
              <span className="font-medium">{metrics?.resources.total || '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Critical Paths</span>
              <span className="font-medium">{metrics?.security.critical_paths || '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Compliance Findings</span>
              <span className="font-medium">{metrics?.compliance.findings || '--'}</span>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            Recent Activity
          </h3>
          <div className="space-y-3">
            <ActivityItem
              type="success"
              message="VM rightsizing completed"
              time="2 minutes ago"
            />
            <ActivityItem
              type="warning"
              message="Cost anomaly detected in Storage"
              time="15 minutes ago"
            />
            <ActivityItem
              type="info"
              message="Compliance scan initiated"
              time="1 hour ago"
            />
            <ActivityItem
              type="success"
              message="Security patch applied"
              time="3 hours ago"
            />
          </div>
        </div>

        {/* AI Learning Status */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            AI Learning Status
          </h3>
          <div className="space-y-4">
            <ProgressBar
              label="Policy Patterns"
              value={87}
              color="blue"
            />
            <ProgressBar
              label="Cost Optimization"
              value={92}
              color="green"
            />
            <ProgressBar
              label="Security Threats"
              value={78}
              color="red"
            />
            <ProgressBar
              label="Compliance Rules"
              value={95}
              color="purple"
            />
          </div>
        </div>
      </div>

      {/* Action Drawer */}
      <ActionDrawer isOpen={isActionDrawerOpen} onClose={() => setIsActionDrawerOpen(false)} actionId={selectedAction} />
    </div>
  );
}

// Helper components
function ActivityItem({ type, message, time }: { type: string; message: string; time: string }) {
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <div className="w-2 h-2 bg-green-500 rounded-full" />;
      case 'warning':
        return <div className="w-2 h-2 bg-yellow-500 rounded-full" />;
      case 'error':
        return <div className="w-2 h-2 bg-red-500 rounded-full" />;
      default:
        return <div className="w-2 h-2 bg-blue-500 rounded-full" />;
    }
  };

  return (
    <div className="flex items-start gap-3">
      <div className="mt-1.5">{getIcon()}</div>
      <div className="flex-1">
        <p className="text-sm text-gray-900 dark:text-white">{message}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400">{time}</p>
      </div>
    </div>
  );
}

function ProgressBar({ label, value, color }: { label: string; value: number; color: string }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
  };

  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
        <span className="text-sm font-medium">{value}%</span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div
          className={`${colorClasses[color as keyof typeof colorClasses]} h-2 rounded-full transition-all duration-300`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}