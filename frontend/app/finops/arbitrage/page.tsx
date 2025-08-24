'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Cloud,
  Zap,
  TrendingDown,
  TrendingUp,
  DollarSign,
  ArrowRight,
  RefreshCw,
  Settings,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  PieChart,
  Activity,
  Sparkles,
  Target,
  Globe,
  Calculator,
  Clock,
  Download,
  Brain
} from 'lucide-react';
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';
import { MLPredictionEngine, PredictionResult } from '@/lib/ml-predictions';

interface ArbitrageOpportunity {
  id: string;
  workloadName: string;
  currentProvider: 'aws' | 'azure' | 'gcp';
  recommendedProvider: 'aws' | 'azure' | 'gcp';
  currentCost: number;
  recommendedCost: number;
  monthlySavings: number;
  migrationComplexity: 'low' | 'medium' | 'high';
  confidenceScore: number;
  requirements: string[];
  estimatedMigrationTime: string;
  businessImpact: 'minimal' | 'moderate' | 'significant';
}

interface CloudPricing {
  provider: 'aws' | 'azure' | 'gcp';
  compute: number;
  storage: number;
  network: number;
  database: number;
  aiml: number;
  total: number;
}

export default function MultiCloudArbitragePage() {
  const router = useRouter();
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [cloudPricing, setCloudPricing] = useState<CloudPricing[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedOpportunity, setSelectedOpportunity] = useState<string | null>(null);
  const [totalPotentialSavings, setTotalPotentialSavings] = useState(0);

  useEffect(() => {
    loadArbitrageData();
  }, []);

  const loadArbitrageData = async () => {
    // Load ML predictions for cost optimization
    const costPrediction = await MLPredictionEngine.predictCostSpike('multi-cloud-arbitrage');
    setPredictions([costPrediction]);

    // Mock arbitrage opportunities
    const mockOpportunities: ArbitrageOpportunity[] = [
      {
        id: 'opp-1',
        workloadName: 'ML Training Pipeline',
        currentProvider: 'aws',
        recommendedProvider: 'gcp',
        currentCost: 12500,
        recommendedCost: 8750,
        monthlySavings: 3750,
        migrationComplexity: 'low',
        confidenceScore: 94,
        requirements: ['GPU compute', 'Large memory', 'Fast storage'],
        estimatedMigrationTime: '2-3 weeks',
        businessImpact: 'minimal'
      },
      {
        id: 'opp-2',
        workloadName: 'Data Analytics Cluster',
        currentProvider: 'azure',
        recommendedProvider: 'aws',
        currentCost: 8200,
        recommendedCost: 6150,
        monthlySavings: 2050,
        migrationComplexity: 'medium',
        confidenceScore: 87,
        requirements: ['Big data processing', 'Real-time analytics', 'High availability'],
        estimatedMigrationTime: '4-6 weeks',
        businessImpact: 'moderate'
      },
      {
        id: 'opp-3',
        workloadName: 'Web Application Backend',
        currentProvider: 'gcp',
        recommendedProvider: 'azure',
        currentCost: 4600,
        recommendedCost: 3450,
        monthlySavings: 1150,
        migrationComplexity: 'low',
        confidenceScore: 91,
        requirements: ['Auto-scaling', 'Load balancing', 'CDN integration'],
        estimatedMigrationTime: '1-2 weeks',
        businessImpact: 'minimal'
      },
      {
        id: 'opp-4',
        workloadName: 'Database Cluster',
        currentProvider: 'aws',
        recommendedProvider: 'gcp',
        currentCost: 15800,
        recommendedCost: 11850,
        monthlySavings: 3950,
        migrationComplexity: 'high',
        confidenceScore: 82,
        requirements: ['High IOPS', 'Multi-AZ deployment', 'Automated backups'],
        estimatedMigrationTime: '8-12 weeks',
        businessImpact: 'significant'
      },
      {
        id: 'opp-5',
        workloadName: 'Development Environment',
        currentProvider: 'azure',
        recommendedProvider: 'aws',
        currentCost: 3200,
        recommendedCost: 2240,
        monthlySavings: 960,
        migrationComplexity: 'low',
        confidenceScore: 89,
        requirements: ['Dev tools integration', 'CI/CD compatibility', 'Testing frameworks'],
        estimatedMigrationTime: '1-2 weeks',
        businessImpact: 'minimal'
      }
    ];

    setOpportunities(mockOpportunities);

    // Calculate total potential savings
    const totalSavings = mockOpportunities.reduce((sum, opp) => sum + opp.monthlySavings, 0);
    setTotalPotentialSavings(totalSavings);

    // Mock cloud pricing comparison
    setCloudPricing([
      {
        provider: 'aws',
        compute: 45000,
        storage: 12000,
        network: 8500,
        database: 22000,
        aiml: 18000,
        total: 105500
      },
      {
        provider: 'azure',
        compute: 42000,
        storage: 11500,
        network: 7800,
        database: 24000,
        aiml: 16500,
        total: 101800
      },
      {
        provider: 'gcp',
        compute: 38000,
        storage: 10800,
        network: 7200,
        database: 19500,
        aiml: 15000,
        total: 90500
      }
    ]);

    setLoading(false);
  };

  const getProviderColor = (provider: string) => {
    switch (provider) {
      case 'aws': return 'orange';
      case 'azure': return 'blue';
      case 'gcp': return 'green';
      default: return 'gray';
    }
  };

  const getProviderIcon = (provider: string) => {
    return <Cloud className="h-5 w-5" />;
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400';
      case 'medium': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'high': return 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400';
      default: return 'text-gray-600 bg-gray-50 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const metrics = [
    {
      id: 'total-savings',
      title: 'Total Monthly Savings',
      value: `$${totalPotentialSavings.toLocaleString()}`,
      change: 28.5,
      trend: 'up' as const,
      sparklineData: [8000, 9200, 10500, 11800, 12100, totalPotentialSavings],
      alert: `${opportunities.length} opportunities identified`
    },
    {
      id: 'active-opportunities',
      title: 'Active Opportunities',
      value: opportunities.length,
      change: 12.0,
      trend: 'up' as const,
      sparklineData: [3, 4, 5, 6, 5, opportunities.length]
    },
    {
      id: 'avg-savings',
      title: 'Average Savings per Workload',
      value: `$${Math.round(totalPotentialSavings / opportunities.length || 0).toLocaleString()}`,
      change: -5.2,
      trend: 'down' as const,
      sparklineData: [2800, 2650, 2400, 2200, 2100, Math.round(totalPotentialSavings / opportunities.length || 0)]
    },
    {
      id: 'confidence-score',
      title: 'Average Confidence',
      value: `${Math.round(opportunities.reduce((sum, opp) => sum + opp.confidenceScore, 0) / opportunities.length || 0)}%`,
      change: 3.8,
      trend: 'up' as const,
      sparklineData: [85, 87, 88, 89, 90, Math.round(opportunities.reduce((sum, opp) => sum + opp.confidenceScore, 0) / opportunities.length || 0)]
    }
  ];

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Zap className="h-10 w-10 text-purple-600" />
            Multi-Cloud Cost Arbitrage
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            AI-powered workload placement optimization across AWS, Azure, and GCP
          </p>
        </div>
        <div className="flex gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => loadArbitrageData()}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <RefreshCw className="h-5 w-5" />
          </button>
          <button
            onClick={() => router.push('/finops/settings')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <Settings className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* AI Predictions Alert */}
      {predictions.length > 0 && predictions[0].riskLevel === 'high' && (
        <div className="mb-6 p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-purple-600 dark:text-purple-400 mt-1" />
            <div className="flex-1">
              <h3 className="font-semibold text-purple-900 dark:text-purple-100">
                AI Arbitrage Prediction
              </h3>
              <p className="text-purple-700 dark:text-purple-300 mt-1">
                New arbitrage opportunities detected based on recent pricing changes
              </p>
              <button className="mt-2 px-3 py-1 bg-purple-600 text-white rounded-md text-sm hover:bg-purple-700">
                Analyze Opportunities
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.id}
            title={metric.title}
            value={metric.value}
            change={metric.change}
            trend={metric.trend}
            sparklineData={metric.sparklineData}
            alert={metric.alert}
          />
        ))}
      </div>

      {view === 'cards' ? (
        <>
          {/* Arbitrage Opportunities */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Target className="h-6 w-6 text-purple-600" />
              Active Arbitrage Opportunities
            </h2>
            <div className="space-y-4">
              {opportunities.map((opportunity) => (
                <div
                  key={opportunity.id}
                  className={`border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-all ${
                    selectedOpportunity === opportunity.id ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20' : ''
                  }`}
                  onClick={() => setSelectedOpportunity(selectedOpportunity === opportunity.id ? null : opportunity.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-lg">{opportunity.workloadName}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(opportunity.migrationComplexity)}`}>
                          {opportunity.migrationComplexity} complexity
                        </span>
                        <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-full text-xs font-semibold">
                          {opportunity.confidenceScore}% confident
                        </span>
                      </div>
                      <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400">
                        <div className="flex items-center gap-2">
                          <span>From:</span>
                          <div className={`flex items-center gap-1 px-2 py-1 rounded bg-${getProviderColor(opportunity.currentProvider)}-50 dark:bg-${getProviderColor(opportunity.currentProvider)}-900/20`}>
                            {getProviderIcon(opportunity.currentProvider)}
                            <span className={`text-${getProviderColor(opportunity.currentProvider)}-700 dark:text-${getProviderColor(opportunity.currentProvider)}-300 font-medium`}>
                              {opportunity.currentProvider.toUpperCase()}
                            </span>
                          </div>
                        </div>
                        <ArrowRight className="h-4 w-4 text-gray-400" />
                        <div className="flex items-center gap-2">
                          <span>To:</span>
                          <div className={`flex items-center gap-1 px-2 py-1 rounded bg-${getProviderColor(opportunity.recommendedProvider)}-50 dark:bg-${getProviderColor(opportunity.recommendedProvider)}-900/20`}>
                            {getProviderIcon(opportunity.recommendedProvider)}
                            <span className={`text-${getProviderColor(opportunity.recommendedProvider)}-700 dark:text-${getProviderColor(opportunity.recommendedProvider)}-300 font-medium`}>
                              {opportunity.recommendedProvider.toUpperCase()}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        ${opportunity.monthlySavings.toLocaleString()}/mo
                      </div>
                      <div className="text-sm text-gray-500">
                        ${opportunity.currentCost.toLocaleString()} → ${opportunity.recommendedCost.toLocaleString()}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Migration: {opportunity.estimatedMigrationTime}
                      </div>
                    </div>
                  </div>

                  {selectedOpportunity === opportunity.id && (
                    <div className="mt-4 pt-4 border-t dark:border-gray-700">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium mb-2">Requirements:</h4>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            {opportunity.requirements.map((req, idx) => (
                              <li key={idx} className="flex items-center gap-2">
                                <CheckCircle className="h-4 w-4 text-green-500" />
                                {req}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium mb-2">Migration Details:</h4>
                          <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <p><span className="font-medium">Time:</span> {opportunity.estimatedMigrationTime}</p>
                            <p><span className="font-medium">Business Impact:</span> {opportunity.businessImpact}</p>
                            <p><span className="font-medium">Complexity:</span> {opportunity.migrationComplexity}</p>
                          </div>
                          <div className="flex gap-2 mt-3">
                            <button className="px-3 py-1 bg-purple-600 text-white rounded-md text-sm hover:bg-purple-700">
                              Start Migration
                            </button>
                            <button className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm hover:bg-gray-200 dark:hover:bg-gray-600">
                              Analyze Further
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Cloud Provider Cost Comparison */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="h-6 w-6 text-blue-600" />
              Cloud Provider Cost Comparison
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {cloudPricing.map((pricing) => (
                <div
                  key={pricing.provider}
                  className={`border-2 rounded-lg p-4 ${
                    pricing.total === Math.min(...cloudPricing.map(p => p.total))
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      {getProviderIcon(pricing.provider)}
                      <h3 className="font-semibold text-lg">{pricing.provider.toUpperCase()}</h3>
                    </div>
                    {pricing.total === Math.min(...cloudPricing.map(p => p.total)) && (
                      <span className="px-2 py-1 bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded-full text-xs font-semibold">
                        Best Value
                      </span>
                    )}
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Compute</span>
                      <span className="font-medium">${pricing.compute.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Storage</span>
                      <span className="font-medium">${pricing.storage.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Network</span>
                      <span className="font-medium">${pricing.network.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Database</span>
                      <span className="font-medium">${pricing.database.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">AI/ML</span>
                      <span className="font-medium">${pricing.aiml.toLocaleString()}</span>
                    </div>
                    <div className="border-t dark:border-gray-700 pt-2 mt-2">
                      <div className="flex justify-between">
                        <span className="font-semibold">Total</span>
                        <span className="font-bold text-lg">${pricing.total.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Visualization Mode */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartContainer
              title="Cost Savings by Provider Migration"
              onDrillIn={() => router.push('/finops/arbitrage/analysis')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Provider migration savings chart visualization</p>
                </div>
              </div>
            </ChartContainer>
            <ChartContainer
              title="Migration Complexity vs Savings"
              onDrillIn={() => router.push('/finops/arbitrage/complexity')}
            >
              <div className="p-4">
                <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                  <p className="text-gray-500">Complexity vs savings scatter plot</p>
                </div>
              </div>
            </ChartContainer>
          </div>

          {/* AI Recommendations */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="h-6 w-6 text-purple-600" />
              AI-Powered Migration Recommendations
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-2">Immediate Actions</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Migrate ML Training to GCP (Save $3,750/mo)
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Move Web Backend to Azure (Save $1,150/mo)
                  </li>
                  <li className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-yellow-500" />
                    Evaluate Dev Environment migration (Save $960/mo)
                  </li>
                </ul>
              </div>
              <div className="p-4 border dark:border-gray-700 rounded-lg">
                <h3 className="font-semibold mb-2">Strategic Planning</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    Database migration requires careful planning
                  </li>
                  <li className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-blue-500" />
                    Analytics cluster shows moderate complexity
                  </li>
                  <li className="flex items-center gap-2">
                    <Calculator className="h-4 w-4 text-purple-500" />
                    Total annual savings potential: $128,880
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}