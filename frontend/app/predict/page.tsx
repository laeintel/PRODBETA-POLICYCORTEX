'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Brain, TrendingUp, AlertTriangle, Shield, Clock,
  GitBranch, ChevronRight, Filter, Download, RefreshCw,
  Target, Zap, Activity, ArrowRight, CheckCircle,
  XCircle, Info, AlertCircle, BarChart3, Eye
} from 'lucide-react';
import RightDrawer, { DrawerSection, DrawerField, DrawerJSON } from '@/components/RightDrawer';
import StatusChip, { RiskChip, ComplianceChip } from '@/components/StatusChip';
import KPICard, { KPICardGrid } from '@/components/KPICard';

interface Prediction {
  id: string;
  type: 'compliance_drift' | 'security_risk' | 'cost_anomaly' | 'performance_degradation';
  title: string;
  description: string;
  confidence: number;
  timeframe: string;
  impact: 'critical' | 'high' | 'medium' | 'low';
  factors: string[];
  recommendation: string;
  autoFixAvailable: boolean;
  riskScore: number;
  affectedResources: string[];
  estimatedCost?: string;
  timestamp: Date;
}

export default function PredictPage() {
  const router = useRouter();
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [filteredPredictions, setFilteredPredictions] = useState<Prediction[]>([]);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState<string>('all');
  const [stats, setStats] = useState({
    total: 0,
    critical: 0,
    highConfidence: 0,
    autoFixable: 0,
    avgConfidence: 0,
    nextETA: ''
  });

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      const res = await fetch('/api/v1/predictions', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        const formattedPredictions: Prediction[] = (data.predictions || []).map((p: any, idx: number) => ({
          id: p.id || `pred-${idx}`,
          type: p.type || 'compliance_drift',
          title: p.title || `${p.resource_type || 'Resource'} - ${p.policy || 'Policy'} Drift`,
          description: p.description || `Predicted ${p.drift_type || 'compliance'} drift with ${Math.round((p.drift_probability || 0.85) * 100)}% probability`,
          confidence: p.confidence || p.drift_probability || 0.85,
          timeframe: p.timeframe || p.predicted_drift_date || '7 days',
          impact: p.impact || (p.drift_probability > 0.8 ? 'high' : 'medium'),
          factors: p.factors || p.contributing_factors || ['Historical patterns', 'Recent changes'],
          recommendation: p.recommendation || 'Review and apply recommended policy adjustments',
          autoFixAvailable: p.auto_fix_available !== false,
          riskScore: p.risk_score || Math.round((p.drift_probability || 0.85) * 100),
          affectedResources: p.affected_resources || [p.resource_id || 'Multiple resources'],
          estimatedCost: p.estimated_cost,
          timestamp: new Date(p.timestamp || p.prediction_timestamp || Date.now())
        }));
        
        setPredictions(formattedPredictions);
        setFilteredPredictions(formattedPredictions);
        calculateStats(formattedPredictions);
      }
    } catch (error) {
      console.error('Failed to load predictions:', error);
      // Load mock data as fallback
      loadMockPredictions();
    } finally {
      setLoading(false);
    }
  };

  const loadMockPredictions = () => {
    const mockPredictions: Prediction[] = [
      {
        id: 'pred-1',
        type: 'compliance_drift',
        title: 'HIPAA Compliance Drift - Patient Data Storage',
        description: 'Predicted compliance violation in 5 days due to unencrypted data at rest',
        confidence: 0.92,
        timeframe: '5 days',
        impact: 'critical',
        factors: ['Encryption policy changes', 'New storage volumes added', 'Missing tags'],
        recommendation: 'Enable encryption on storage accounts and update tagging policy',
        autoFixAvailable: true,
        riskScore: 92,
        affectedResources: ['/subscriptions/205b477d/storage/patient-data-01'],
        estimatedCost: '$15,000',
        timestamp: new Date()
      },
      {
        id: 'pred-2',
        type: 'cost_anomaly',
        title: 'Unusual Compute Spend Trajectory',
        description: 'Projected 45% cost increase by month end based on current usage patterns',
        confidence: 0.87,
        timeframe: '14 days',
        impact: 'high',
        factors: ['Increased VM usage', 'Premium tier migrations', 'Orphaned resources'],
        recommendation: 'Right-size VMs and implement auto-shutdown policies',
        autoFixAvailable: true,
        riskScore: 78,
        affectedResources: ['/subscriptions/205b477d/vms/*'],
        estimatedCost: '$45,000',
        timestamp: new Date()
      },
      {
        id: 'pred-3',
        type: 'security_risk',
        title: 'Potential Privilege Escalation Path',
        description: 'Excessive permissions detected that could lead to unauthorized access',
        confidence: 0.89,
        timeframe: '3 days',
        impact: 'high',
        factors: ['Role assignments', 'Service principal permissions', 'Group memberships'],
        recommendation: 'Apply principle of least privilege and review role assignments',
        autoFixAvailable: true,
        riskScore: 85,
        affectedResources: ['/subscriptions/205b477d/rbac/roles/*'],
        timestamp: new Date()
      }
    ];
    
    setPredictions(mockPredictions);
    setFilteredPredictions(mockPredictions);
    calculateStats(mockPredictions);
  };

  const calculateStats = (preds: Prediction[]) => {
    const critical = preds.filter(p => p.impact === 'critical').length;
    const highConf = preds.filter(p => p.confidence > 0.85).length;
    const autoFix = preds.filter(p => p.autoFixAvailable).length;
    const avgConf = preds.length > 0 
      ? preds.reduce((sum, p) => sum + p.confidence, 0) / preds.length 
      : 0;
    
    // Find next ETA
    const sortedByTime = [...preds].sort((a, b) => {
      const getDays = (timeframe: string) => {
        const match = timeframe.match(/(\d+)/);
        return match ? parseInt(match[1]) : 999;
      };
      return getDays(a.timeframe) - getDays(b.timeframe);
    });
    
    setStats({
      total: preds.length,
      critical,
      highConfidence: highConf,
      autoFixable: autoFix,
      avgConfidence: avgConf,
      nextETA: sortedByTime[0]?.timeframe || 'N/A'
    });
  };

  useEffect(() => {
    let filtered = [...predictions];
    
    if (filterType !== 'all') {
      filtered = filtered.filter(p => p.type === filterType);
    }
    
    setFilteredPredictions(filtered);
  }, [filterType, predictions]);

  const handleCreateFixPR = (prediction: Prediction) => {
    // In production, this would call an API to create a PR
    window.open(`https://github.com/your-org/repo/pulls/new?title=Fix: ${encodeURIComponent(prediction.title)}`, '_blank');
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'compliance_drift': return <Shield className="w-4 h-4" />;
      case 'security_risk': return <AlertTriangle className="w-4 h-4" />;
      case 'cost_anomaly': return <TrendingUp className="w-4 h-4" />;
      case 'performance_degradation': return <Activity className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical': return 'text-red-600 dark:text-red-400';
      case 'high': return 'text-orange-600 dark:text-orange-400';
      case 'medium': return 'text-yellow-600 dark:text-yellow-400';
      case 'low': return 'text-green-600 dark:text-green-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground dark:text-white flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              Predict
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground dark:text-gray-400 mt-1">
              AI predictions with 7-day look-ahead and auto-remediation
            </p>
          </div>
          
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => loadPredictions()}
              className="px-4 py-2 bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <button
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
              onClick={() => router.push('/ai/models')}
            >
              <BarChart3 className="w-4 h-4" />
              Model Insights
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <KPICardGrid columns={4}>
        <KPICard
          title="Active Predictions"
          metric={stats.total}
          subtext="Across all categories"
          icon={<Brain />}
          trend={stats.total > 0 ? 'up' : 'stable'}
          change={12}
          action={{
            label: 'View All',
            onClick: () => setFilterType('all')
          }}
        />
        <KPICard
          title="Critical Impact"
          metric={stats.critical}
          subtext="Requiring immediate action"
          icon={<AlertTriangle />}
          trend={stats.critical > 0 ? 'up' : 'stable'}
          trendColor="negative"
          variant="gradient"
        />
        <KPICard
          title="Model Confidence"
          metric={`${Math.round(stats.avgConfidence * 100)}%`}
          subtext="Average prediction confidence"
          icon={<Target />}
          sparklineData={[85, 87, 89, 92, 90, 93, 92]}
        />
        <KPICard
          title="Auto-Fix Available"
          metric={stats.autoFixable}
          subtext={`Next ETA: ${stats.nextETA}`}
          icon={<Zap />}
          action={{
            label: 'Create Fix PRs',
            onClick: () => {}
          }}
        />
      </KPICardGrid>

      {/* Filter Controls */}
      <div className="mt-6 mb-4 flex flex-wrap gap-2">
        <button
          onClick={() => setFilterType('all')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            filterType === 'all'
              ? 'bg-purple-600 text-white'
              : 'bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-700'
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilterType('compliance_drift')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            filterType === 'compliance_drift'
              ? 'bg-purple-600 text-white'
              : 'bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-700'
          }`}
        >
          Compliance Drift
        </button>
        <button
          onClick={() => setFilterType('security_risk')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            filterType === 'security_risk'
              ? 'bg-purple-600 text-white'
              : 'bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-700'
          }`}
        >
          Security Risk
        </button>
        <button
          onClick={() => setFilterType('cost_anomaly')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            filterType === 'cost_anomaly'
              ? 'bg-purple-600 text-white'
              : 'bg-muted dark:bg-gray-800 text-muted-foreground dark:text-gray-300 hover:bg-accent dark:hover:bg-gray-700'
          }`}
        >
          Cost Anomaly
        </button>
      </div>

      {/* Predictions List */}
      <div className="space-y-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 animate-spin text-purple-600 dark:text-purple-400" />
          </div>
        ) : filteredPredictions.length === 0 ? (
          <div className="text-center py-12 bg-card dark:bg-gray-800 rounded-lg">
            <Brain className="w-12 h-12 mx-auto text-muted-foreground dark:text-gray-400 mb-4" />
            <p className="text-muted-foreground dark:text-gray-400">No predictions match your filter</p>
          </div>
        ) : (
          filteredPredictions.map((prediction) => (
            <div
              key={prediction.id}
              className="bg-card dark:bg-gray-800 rounded-lg p-6 hover:shadow-lg transition-all cursor-pointer border border-border dark:border-gray-700"
              onClick={() => {
                setSelectedPrediction(prediction);
                setDrawerOpen(true);
              }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      prediction.type === 'compliance_drift' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' :
                      prediction.type === 'security_risk' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' :
                      prediction.type === 'cost_anomaly' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400' :
                      'bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400'
                    }`}>
                      {getTypeIcon(prediction.type)}
                    </div>
                    
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-foreground dark:text-white mb-1">
                        {prediction.title}
                      </h3>
                      <p className="text-sm text-muted-foreground dark:text-gray-400 mb-3">
                        {prediction.description}
                      </p>
                      
                      <div className="flex flex-wrap items-center gap-4 text-sm">
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4 text-muted-foreground dark:text-gray-400" />
                          <span className="font-medium">{prediction.timeframe}</span>
                        </div>
                        
                        <div className="flex items-center gap-1">
                          <Target className="w-4 h-4 text-muted-foreground dark:text-gray-400" />
                          <span className="font-medium">{Math.round(prediction.confidence * 100)}% confidence</span>
                        </div>
                        
                        <div className={`flex items-center gap-1 ${getImpactColor(prediction.impact)}`}>
                          <AlertCircle className="w-4 h-4" />
                          <span className="font-medium capitalize">{prediction.impact} impact</span>
                        </div>
                        
                        {prediction.estimatedCost && (
                          <div className="flex items-center gap-1 text-orange-600 dark:text-orange-400">
                            <TrendingUp className="w-4 h-4" />
                            <span className="font-medium">{prediction.estimatedCost}</span>
                          </div>
                        )}
                      </div>
                      
                      <div className="mt-3 flex items-center gap-2">
                        <RiskChip score={prediction.riskScore} size="sm" />
                        {prediction.autoFixAvailable && (
                          <StatusChip variant="success" label="Auto-Fix Available" size="sm" showIcon />
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col gap-2 ml-4">
                  {prediction.autoFixAvailable && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCreateFixPR(prediction);
                      }}
                      className="px-3 py-1.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2 text-sm"
                    >
                      <GitBranch className="w-4 h-4" />
                      Create Fix PR
                    </button>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedPrediction(prediction);
                      setDrawerOpen(true);
                    }}
                    className="px-3 py-1.5 bg-muted dark:bg-gray-700 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-600 transition-colors flex items-center gap-2 text-sm"
                  >
                    <Eye className="w-4 h-4" />
                    View Details
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Detail Drawer */}
      <RightDrawer
        isOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        title={selectedPrediction?.title || 'Prediction Details'}
        subtitle={`${Math.round((selectedPrediction?.confidence || 0) * 100)}% confidence • ETA: ${selectedPrediction?.timeframe}`}
        width="lg"
        footer={
          selectedPrediction?.autoFixAvailable && (
            <div className="flex gap-3">
              <button
                onClick={() => handleCreateFixPR(selectedPrediction)}
                className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
              >
                <GitBranch className="w-4 h-4" />
                Create Fix PR
              </button>
              <button
                onClick={() => router.push(`/resources?filter=${encodeURIComponent(selectedPrediction.affectedResources[0])}`)}
                className="flex-1 px-4 py-2 bg-muted dark:bg-gray-700 text-muted-foreground dark:text-gray-300 rounded-lg hover:bg-accent dark:hover:bg-gray-600 transition-colors flex items-center justify-center gap-2"
              >
                View Resources
              </button>
            </div>
          )
        }
      >
        {selectedPrediction && (
          <>
            <DrawerSection title="Prediction Overview">
              <p className="text-sm text-muted-foreground dark:text-gray-400 mb-4">
                {selectedPrediction.description}
              </p>
              <div className="space-y-2">
                <DrawerField label="Type" value={selectedPrediction.type.replace('_', ' ')} />
                <DrawerField label="Impact" value={
                  <span className={getImpactColor(selectedPrediction.impact)}>
                    {selectedPrediction.impact.toUpperCase()}
                  </span>
                } />
                <DrawerField label="Risk Score" value={`${selectedPrediction.riskScore}%`} />
                <DrawerField label="Timeframe" value={selectedPrediction.timeframe} />
                <DrawerField label="Confidence" value={`${Math.round(selectedPrediction.confidence * 100)}%`} />
              </div>
            </DrawerSection>

            <DrawerSection title="Contributing Factors">
              <ul className="space-y-2">
                {selectedPrediction.factors.map((factor, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-purple-600 dark:text-purple-400 mt-0.5" />
                    <span className="text-muted-foreground dark:text-gray-300">{factor}</span>
                  </li>
                ))}
              </ul>
            </DrawerSection>

            <DrawerSection title="Recommendation">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <p className="text-sm text-blue-900 dark:text-blue-100">
                  {selectedPrediction.recommendation}
                </p>
              </div>
            </DrawerSection>

            <DrawerSection title="Affected Resources">
              <div className="space-y-2">
                {selectedPrediction.affectedResources.map((resource, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-muted dark:bg-gray-900 rounded">
                    <code className="text-xs">{resource}</code>
                    <button
                      onClick={() => router.push(`/resources?filter=${encodeURIComponent(resource)}`)}
                      className="text-blue-600 dark:text-blue-400 hover:underline text-xs"
                    >
                      View
                    </button>
                  </div>
                ))}
              </div>
            </DrawerSection>

            {selectedPrediction.estimatedCost && (
              <DrawerSection title="Financial Impact">
                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg">
                  <p className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                    {selectedPrediction.estimatedCost}
                  </p>
                  <p className="text-sm text-orange-700 dark:text-orange-300 mt-1">
                    Estimated cost impact if not addressed
                  </p>
                </div>
              </DrawerSection>
            )}
          </>
        )}
      </RightDrawer>
    </div>
  );
}