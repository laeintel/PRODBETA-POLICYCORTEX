'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Target,
  TrendingDown,
  DollarSign,
  Cpu,
  HardDrive,
  Network,
  Zap,
  CheckCircle,
  AlertTriangle,
  ArrowLeft,
  Play,
  Pause,
  RefreshCw,
  Download,
  Brain,
  Sparkles,
  Clock,
  Server
} from 'lucide-react';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface OptimizationRecommendation {
  id: string;
  resource: string;
  resourceType: 'compute' | 'storage' | 'network' | 'database' | 'container';
  currentCost: number;
  optimizedCost: number;
  savings: number;
  savingsPercent: number;
  recommendation: string;
  effort: 'low' | 'medium' | 'high';
  risk: 'low' | 'medium' | 'high';
  confidence: number;
  automationAvailable: boolean;
  impact: string;
  steps: string[];
}

export default function AutomatedOptimizationPage() {
  const router = useRouter();
  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState<OptimizationRecommendation | null>(null);
  const [autoOptimizeEnabled, setAutoOptimizeEnabled] = useState(false);
  const [optimizing, setOptimizing] = useState<string | null>(null);
  const [totalSavings, setTotalSavings] = useState(0);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);

  useEffect(() => {
    // Load ML predictions
    const loadPredictions = async () => {
      const wastePrediction = await MLPredictionEngine.predictResourceWaste('all-resources');
      setPredictions([wastePrediction]);
    };
    loadPredictions();

    // Generate optimization recommendations
    const recs: OptimizationRecommendation[] = [
      {
        id: 'opt-1',
        resource: 'prod-web-servers',
        resourceType: 'compute',
        currentCost: 45000,
        optimizedCost: 28000,
        savings: 17000,
        savingsPercent: 38,
        recommendation: 'Rightsize EC2 instances from m5.4xlarge to m5.2xlarge',
        effort: 'low',
        risk: 'low',
        confidence: 94,
        automationAvailable: true,
        impact: 'CPU utilization averaging 23%, can safely downsize',
        steps: [
          'Create AMI backup of current instances',
          'Launch new m5.2xlarge instances',
          'Test application performance',
          'Switch traffic to new instances',
          'Terminate old instances'
        ]
      },
      {
        id: 'opt-2',
        resource: 'data-warehouse-cluster',
        resourceType: 'database',
        currentCost: 32000,
        optimizedCost: 18000,
        savings: 14000,
        savingsPercent: 44,
        recommendation: 'Convert RDS to Aurora Serverless v2 with auto-pause',
        effort: 'medium',
        risk: 'medium',
        confidence: 87,
        automationAvailable: true,
        impact: 'Database idle 60% of time, serverless will auto-scale',
        steps: [
          'Create Aurora Serverless v2 cluster',
          'Migrate data using DMS',
          'Update application connection strings',
          'Monitor performance for 24 hours',
          'Decommission old RDS instance'
        ]
      },
      {
        id: 'opt-3',
        resource: 'backup-storage-s3',
        resourceType: 'storage',
        currentCost: 12000,
        optimizedCost: 4800,
        savings: 7200,
        savingsPercent: 60,
        recommendation: 'Move old backups to Glacier Deep Archive',
        effort: 'low',
        risk: 'low',
        confidence: 98,
        automationAvailable: true,
        impact: 'Backups older than 90 days rarely accessed',
        steps: [
          'Identify backups older than 90 days',
          'Create lifecycle policy for automatic tiering',
          'Move identified objects to Glacier',
          'Update backup retention policy'
        ]
      },
      {
        id: 'opt-4',
        resource: 'dev-environment-vms',
        resourceType: 'compute',
        currentCost: 8500,
        optimizedCost: 2100,
        savings: 6400,
        savingsPercent: 75,
        recommendation: 'Implement auto-shutdown for dev/test environments',
        effort: 'low',
        risk: 'low',
        confidence: 96,
        automationAvailable: true,
        impact: 'Dev VMs running 24/7 but only used 8 hours/day',
        steps: [
          'Tag all dev/test resources',
          'Create auto-shutdown schedule (7 PM - 7 AM)',
          'Implement weekend shutdown policy',
          'Set up on-demand start capability'
        ]
      },
      {
        id: 'opt-5',
        resource: 'kubernetes-cluster',
        resourceType: 'container',
        currentCost: 22000,
        optimizedCost: 15000,
        savings: 7000,
        savingsPercent: 32,
        recommendation: 'Enable cluster autoscaling and spot instances',
        effort: 'medium',
        risk: 'medium',
        confidence: 85,
        automationAvailable: true,
        impact: 'Cluster overprovisioned for peak load that occurs 5% of time',
        steps: [
          'Enable Kubernetes cluster autoscaler',
          'Configure spot instance node pools',
          'Set up pod disruption budgets',
          'Implement horizontal pod autoscaling',
          'Monitor and adjust thresholds'
        ]
      },
      {
        id: 'opt-6',
        resource: 'cross-region-transfer',
        resourceType: 'network',
        currentCost: 6300,
        optimizedCost: 2100,
        savings: 4200,
        savingsPercent: 67,
        recommendation: 'Optimize data transfer with CDN and caching',
        effort: 'high',
        risk: 'low',
        confidence: 82,
        automationAvailable: false,
        impact: 'Redundant data transfers between regions',
        steps: [
          'Analyze data transfer patterns',
          'Implement CloudFront CDN',
          'Set up regional caching layers',
          'Optimize application architecture',
          'Monitor transfer costs'
        ]
      }
    ];

    setRecommendations(recs);
    setTotalSavings(recs.reduce((sum, rec) => sum + rec.savings, 0));
  }, []);

  const executeOptimization = async (rec: OptimizationRecommendation) => {
    setOptimizing(rec.id);
    
    // Simulate optimization execution
    setTimeout(() => {
      setOptimizing(null);
      // Update recommendation status
      setRecommendations(prev => 
        prev.map(r => 
          r.id === rec.id 
            ? { ...r, currentCost: r.optimizedCost, savings: 0 }
            : r
        )
      );
    }, 5000);
  };

  const getResourceIcon = (type: string) => {
    switch (type) {
      case 'compute': return <Server className="h-5 w-5" />;
      case 'storage': return <HardDrive className="h-5 w-5" />;
      case 'network': return <Network className="h-5 w-5" />;
      case 'database': return <Cpu className="h-5 w-5" />;
      case 'container': return <Server className="h-5 w-5" />;
      default: return <Server className="h-5 w-5" />;
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-50 dark:bg-green-900/20';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20';
      case 'high': return 'text-red-600 bg-red-50 dark:bg-red-900/20';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20';
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/finops')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Target className="h-8 w-8 text-green-600" />
              Automated Rightsizing & Optimization
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              ML-driven recommendations save ${totalSavings.toLocaleString()}/month
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setAutoOptimizeEnabled(!autoOptimizeEnabled)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              autoOptimizeEnabled
                ? 'bg-green-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            {autoOptimizeEnabled ? (
              <>
                <Zap className="h-4 w-4" />
                Auto-Optimize ON
              </>
            ) : (
              <>
                <Zap className="h-4 w-4" />
                Auto-Optimize OFF
              </>
            )}
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export Report
          </button>
        </div>
      </div>

      {/* ML Prediction Alert */}
      {predictions.length > 0 && (
        <div className="mb-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-amber-600 dark:text-amber-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-amber-900 dark:text-amber-100">
                AI Waste Detection
              </h3>
              <p className="text-amber-700 dark:text-amber-300 mt-1">
                {predictions[0].prediction}
              </p>
              <p className="text-sm text-amber-600 dark:text-amber-400 mt-1">
                {predictions[0].explanation}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Savings</p>
              <p className="text-2xl font-bold text-green-600">
                ${totalSavings.toLocaleString()}/mo
              </p>
            </div>
            <DollarSign className="h-8 w-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Recommendations</p>
              <p className="text-2xl font-bold">{recommendations.length}</p>
            </div>
            <Sparkles className="h-8 w-8 text-purple-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Auto-Executable</p>
              <p className="text-2xl font-bold">
                {recommendations.filter(r => r.automationAvailable).length}
              </p>
            </div>
            <Zap className="h-8 w-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</p>
              <p className="text-2xl font-bold">
                {Math.round(recommendations.reduce((sum, r) => sum + r.confidence, 0) / recommendations.length)}%
              </p>
            </div>
            <Brain className="h-8 w-8 text-blue-500" />
          </div>
        </div>
      </div>

      {/* Recommendations List */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">Optimization Recommendations</h2>
          <div className="space-y-4">
            {recommendations.map((rec) => (
              <div
                key={rec.id}
                className="border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                onClick={() => setSelectedRecommendation(rec)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                      {getResourceIcon(rec.resourceType)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="font-semibold">{rec.resource}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(rec.risk)}`}>
                          {rec.risk.toUpperCase()} RISK
                        </span>
                        <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-full text-xs font-medium">
                          {rec.effort.toUpperCase()} EFFORT
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {rec.recommendation}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500">
                        {rec.impact}
                      </p>
                      <div className="flex items-center gap-4 mt-2">
                        <span className="text-sm">
                          Current: <span className="font-semibold">${rec.currentCost.toLocaleString()}/mo</span>
                        </span>
                        <span className="text-sm">
                          Optimized: <span className="font-semibold text-green-600">${rec.optimizedCost.toLocaleString()}/mo</span>
                        </span>
                        <span className="text-sm font-semibold text-green-600">
                          Save ${rec.savings.toLocaleString()} ({rec.savingsPercent}%)
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <span className="text-sm text-gray-500">
                      {rec.confidence}% confident
                    </span>
                    {rec.automationAvailable && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          executeOptimization(rec);
                        }}
                        disabled={optimizing === rec.id}
                        className="px-4 py-2 bg-green-600 text-white rounded-md text-sm hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
                      >
                        {optimizing === rec.id ? (
                          <>
                            <RefreshCw className="h-4 w-4 animate-spin" />
                            Optimizing...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4" />
                            Auto-Optimize
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Selected Recommendation Details */}
      {selectedRecommendation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Optimization Details</h2>
                <button
                  onClick={() => setSelectedRecommendation(null)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">{selectedRecommendation.resource}</h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {selectedRecommendation.recommendation}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Current Cost</p>
                    <p className="text-xl font-bold">${selectedRecommendation.currentCost.toLocaleString()}/mo</p>
                  </div>
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <p className="text-sm text-green-600 dark:text-green-400">Optimized Cost</p>
                    <p className="text-xl font-bold text-green-600 dark:text-green-400">
                      ${selectedRecommendation.optimizedCost.toLocaleString()}/mo
                    </p>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Implementation Steps</h4>
                  <ol className="list-decimal list-inside space-y-1">
                    {selectedRecommendation.steps.map((step, idx) => (
                      <li key={idx} className="text-sm text-gray-600 dark:text-gray-400">
                        {step}
                      </li>
                    ))}
                  </ol>
                </div>

                <div className="flex gap-3">
                  {selectedRecommendation.automationAvailable && (
                    <button
                      onClick={() => {
                        executeOptimization(selectedRecommendation);
                        setSelectedRecommendation(null);
                      }}
                      className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                    >
                      Execute Auto-Optimization
                    </button>
                  )}
                  <button
                    onClick={() => setSelectedRecommendation(null)}
                    className="flex-1 px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Add missing import
import { X } from 'lucide-react';