'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import {
  Sparkles,
  BarChart3,
  MessageSquare,
  Box,
  ArrowLeft,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Cpu,
  Beaker,
  Lightbulb,
  Terminal,
  TrendingUp,
  ShieldCheck
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import ViewToggle from '@/components/ViewToggle';
import MetricCard from '@/components/MetricCard';
import ChartContainer from '@/components/ChartContainer';
import DataExport from '@/components/DataExport';
import { useViewPreference } from '@/hooks/useViewPreference';

interface AICard {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  stats: {
    label: string;
    value: string | number;
    trend?: 'up' | 'down' | 'stable';
    status?: 'success' | 'warning' | 'error';
  }[];
  route: string;
  color: string;
  patentNumber?: string;
  actions?: { label: string; onClick: () => void }[];
}

export default function AIPage() {
  const router = useRouter();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const { view, setView } = useViewPreference('ai-view', 'cards');

  // Mock data for AI charts
  const modelAccuracyData = [
    { date: '2024-01-01', accuracy: 96.8, predictions: 3200, latency: 92 },
    { date: '2024-01-02', accuracy: 97.2, predictions: 3450, latency: 89 },
    { date: '2024-01-03', accuracy: 97.5, predictions: 3678, latency: 87 },
    { date: '2024-01-04', accuracy: 98.1, predictions: 3847, latency: 85 },
    { date: '2024-01-05', accuracy: 98.7, predictions: 4023, latency: 83 },
    { date: '2024-01-06', accuracy: 99.2, predictions: 4156, latency: 87 }
  ];

  const patentUsageData = [
    { patent: 'Patent #1\nCorrelations', usage: 89, impact: 94 },
    { patent: 'Patent #2\nConversational AI', usage: 76, impact: 91 },
    { patent: 'Patent #3\nUnified Platform', usage: 93, impact: 96 },
    { patent: 'Patent #4\nPredictive Engine', usage: 82, impact: 98 }
  ];

  const inferenceVolume = [
    { name: 'Governance', value: 3847, color: '#3B82F6' },
    { name: 'Security', value: 2341, color: '#EF4444' },
    { name: 'Compliance', value: 1876, color: '#10B981' },
    { name: 'Operations', value: 1234, color: '#F59E0B' }
  ];

  const aiCapabilities = [
    { subject: 'Prediction', score: 99, fullMark: 100 },
    { subject: 'Classification', score: 96, fullMark: 100 },
    { subject: 'Correlation', score: 94, fullMark: 100 },
    { subject: 'NLP Processing', score: 98, fullMark: 100 },
    { subject: 'Pattern Recognition', score: 92, fullMark: 100 },
    { subject: 'Anomaly Detection', score: 97, fullMark: 100 }
  ];

  const aiCards: AICard[] = [
    {
      id: 'predictive',
      title: 'Predictive Compliance Engine',
      description: 'ML-powered policy drift prediction and compliance forecasting',
      icon: TrendingUp,
      stats: [
        { label: 'Accuracy', value: '99.2%', status: 'success' },
        { label: 'Predictions Today', value: '3,847', trend: 'up' },
        { label: 'Drift Detected', value: '42', status: 'warning' },
        { label: 'Avg Latency', value: '87ms', trend: 'down' }
      ],
      route: '/ai/predictive',
      patentNumber: 'Patent #4',
      color: 'purple',
      actions: [
        { label: 'View Predictions', onClick: () => router.push('/ai/predictive') },
        { label: 'Train Models', onClick: () => router.push('/ai/predictive?action=train') }
      ]
    },
    {
      id: 'correlations',
      title: 'Cross-Domain Analysis',
      description: 'Discover hidden patterns across governance domains',
      icon: BarChart3,
      stats: [
        { label: 'Correlations Found', value: '1,234', trend: 'up' },
        { label: 'Domains Analyzed', value: '8', status: 'success' },
        { label: 'Active Patterns', value: '156', trend: 'stable' },
        { label: 'Confidence Score', value: '94.7%', status: 'success' }
      ],
      route: '/ai/correlations',
      patentNumber: 'Patent #1',
      color: 'blue',
      actions: [
        { label: 'View Analysis', onClick: () => router.push('/ai/correlations') },
        { label: 'Run Analysis', onClick: () => router.push('/ai/correlations?action=analyze') }
      ]
    },
    {
      id: 'chat',
      title: 'Conversational AI Interface',
      description: 'Natural language governance management and insights',
      icon: MessageSquare,
      stats: [
        { label: 'Active Sessions', value: '47', trend: 'up' },
        { label: 'Queries Today', value: '892', status: 'success' },
        { label: 'Avg Response', value: '1.2s', trend: 'down' },
        { label: 'Satisfaction', value: '98.3%', status: 'success' }
      ],
      route: '/ai/chat',
      patentNumber: 'Patent #2',
      color: 'green',
      actions: [
        { label: 'Start Chat', onClick: () => router.push('/ai/chat') },
        { label: 'View History', onClick: () => router.push('/ai/chat?tab=history') }
      ]
    },
    {
      id: 'unified',
      title: 'Unified Platform Metrics',
      description: 'Holistic AI-driven cloud governance insights',
      icon: Box,
      stats: [
        { label: 'Models Active', value: '24', status: 'success' },
        { label: 'Data Processed', value: '2.4TB', trend: 'up' },
        { label: 'Insights Generated', value: '5,678', trend: 'up' },
        { label: 'Platform Health', value: '99.8%', status: 'success' }
      ],
      route: '/ai/unified',
      patentNumber: 'Patent #3',
      color: 'orange',
      actions: [
        { label: 'View Dashboard', onClick: () => router.push('/ai/unified') },
        { label: 'Configure AI', onClick: () => router.push('/ai/unified?tab=config') }
      ]
    }
  ];

  const recentActivities = [
    { id: 1, type: 'prediction', message: 'Policy drift detected in production environment', time: '5 min ago', status: 'warning' },
    { id: 2, type: 'training', message: 'Compliance model training completed (99.3% accuracy)', time: '15 min ago', status: 'success' },
    { id: 3, type: 'correlation', message: 'New correlation pattern discovered between IAM and Cost', time: '30 min ago', status: 'info' },
    { id: 4, type: 'chat', message: 'AI assistant resolved 42 governance queries', time: '1 hour ago', status: 'success' },
    { id: 5, type: 'analysis', message: 'Cross-domain analysis completed for Q1 2024', time: '2 hours ago', status: 'success' }
  ];

  const quickStats = [
    { label: 'AI Models', value: '89', icon: Cpu, color: 'purple' },
    { label: 'Accuracy Rate', value: '98.7%', icon: CheckCircle, color: 'green' },
    { label: 'Processing Speed', value: '124ms', icon: Clock, color: 'blue' },
    { label: 'Active Jobs', value: '12', icon: Beaker, color: 'orange' }
  ];

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'success': return 'text-green-600 dark:text-green-400';
      case 'warning': return 'text-yellow-600 dark:text-yellow-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      default: return 'text-blue-600 dark:text-blue-400';
    }
  };

  const getTrendSymbol = (trend?: string) => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'prediction': return TrendingUp;
      case 'training': return Beaker;
      case 'correlation': return BarChart3;
      case 'chat': return MessageSquare;
      case 'analysis': return Box;
      default: return Sparkles;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/tactical')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            aria-label="Back to Command Center"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-4xl font-bold">AI Intelligence Hub</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Patented AI technologies for intelligent cloud governance
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <ViewToggle view={view} onViewChange={setView} />
          <button
            onClick={() => router.push('/ai/chat')}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
          >
            <MessageSquare className="h-5 w-5" />
            AI Assistant
          </button>
          <button
            onClick={() => router.push('/ai/predictive?action=analyze')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Sparkles className="h-5 w-5" />
            Run Analysis
          </button>
        </div>
      </div>

      {/* Patent Notice */}
      <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-xl p-4 mb-8">
        <div className="flex items-center gap-3">
          <ShieldCheck className="h-6 w-6 text-purple-600 dark:text-purple-400" />
          <div>
            <p className="text-sm font-medium text-purple-900 dark:text-purple-100">
              Protected by 4 U.S. Patents
            </p>
            <p className="text-xs text-purple-700 dark:text-purple-300 mt-1">
              This platform implements proprietary AI technologies covered by US Patent Applications 
              17/123,458, 18/234,567, 19/345,678, and 20/456,789
            </p>
          </div>
        </div>
      </div>

      {view === 'cards' ? (
        <>
          {/* Quick Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <MetricCard
              title="AI Models"
              value={89}
              change={8}
              changeLabel="new models"
              icon={<Cpu className="h-5 w-5 text-purple-600" />}
              sparklineData={[81, 83, 85, 87, 88, 89]}
              onClick={() => router.push('/ai/unified')}
              status="success"
            />
            <MetricCard
              title="Accuracy Rate"
              value="98.7%"
              change={0.5}
              changeLabel="improvement"
              icon={<CheckCircle className="h-5 w-5 text-green-600" />}
              sparklineData={[97.8, 98.1, 98.3, 98.5, 98.6, 98.7]}
              onClick={() => router.push('/ai/predictive')}
              status="success"
            />
            <MetricCard
              title="Processing Speed"
              value="124ms"
              change={-12}
              changeLabel="faster"
              icon={<Clock className="h-5 w-5 text-blue-600" />}
              sparklineData={[145, 138, 132, 128, 126, 124]}
              onClick={() => router.push('/ai/unified')}
              status="success"
            />
            <MetricCard
              title="Active Jobs"
              value={12}
              change={20}
              changeLabel="increase"
              icon={<Beaker className="h-5 w-5 text-orange-600" />}
              sparklineData={[8, 9, 10, 11, 12, 12]}
              onClick={() => router.push('/ai/unified')}
              status="neutral"
            />
          </div>
        </>
      ) : (
        <>
          {/* AI Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartContainer 
              title="Model Performance Trends" 
              onExport={() => {}}
              onDrillIn={() => router.push('/ai/predictive')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={modelAccuracyData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="date" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#10B981" strokeWidth={2} name="Accuracy %" />
                  <Line type="monotone" dataKey="predictions" stroke="#3B82F6" strokeWidth={2} name="Daily Predictions" />
                  <Line type="monotone" dataKey="latency" stroke="#F59E0B" strokeWidth={2} name="Latency (ms)" />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="Patent Technology Usage" 
              onExport={() => {}}
              onDrillIn={() => router.push('/ai/unified')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={patentUsageData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-700" />
                  <XAxis dataKey="patent" className="text-gray-600 dark:text-gray-400" />
                  <YAxis className="text-gray-600 dark:text-gray-400" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="usage" fill="#8B5CF6" name="Usage %" />
                  <Bar dataKey="impact" fill="#3B82F6" name="Impact Score" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="AI Inference Volume" 
              onExport={() => {}}
              onDrillIn={() => router.push('/ai/correlations')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={inferenceVolume}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {inferenceVolume.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer 
              title="AI Capability Matrix" 
              onExport={() => {}}
              onDrillIn={() => router.push('/ai/unified')}
            >
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={aiCapabilities}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar
                    name="AI Capabilities"
                    dataKey="score"
                    stroke="#8B5CF6"
                    fill="#8B5CF6"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </>
      )}

      {view === 'cards' && (
        <>
          {/* AI Feature Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {aiCards.map((card) => {
              const Icon = card.icon;
              return (
                <div
                  key={card.id}
                  className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-lg transition-all cursor-pointer transform hover:scale-[1.02]"
                  onMouseEnter={() => setHoveredCard(card.id)}
                  onMouseLeave={() => setHoveredCard(null)}
                  onClick={() => router.push(card.route)}
                >
                  <div className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className={`p-3 rounded-lg bg-${card.color}-50 dark:bg-${card.color}-900/20`}>
                        <Icon className={`h-8 w-8 text-${card.color}-600 dark:text-${card.color}-400`} />
                      </div>
                      <div className="flex items-center gap-2">
                        {card.patentNumber && (
                          <span className="text-xs px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full font-medium">
                            {card.patentNumber}
                          </span>
                        )}
                        {hoveredCard === card.id && (
                          <ArrowLeft className="h-5 w-5 rotate-180 text-gray-400" />
                        )}
                      </div>
                    </div>
                    
                    <h3 className="text-xl font-semibold mb-2">{card.title}</h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                      {card.description}
                    </p>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      {card.stats.map((stat, index) => (
                        <div key={index} className="text-sm">
                          <p className="text-gray-500 dark:text-gray-400">{stat.label}</p>
                          <p className={`font-semibold flex items-center gap-1 ${
                            stat.status ? getStatusColor(stat.status) : ''
                          }`}>
                            {stat.value}
                            {stat.trend && (
                              <span className={`text-xs ${
                                stat.trend === 'up' ? 'text-green-500' :
                                stat.trend === 'down' ? 'text-red-500' :
                                'text-gray-500'
                              }`}>
                                {getTrendSymbol(stat.trend)}
                              </span>
                            )}
                          </p>
                        </div>
                      ))}
                    </div>

                    {/* Action Buttons */}
                    {card.actions && (
                      <div className="flex gap-2 pt-3 border-t dark:border-gray-700">
                        {card.actions.map((action, index) => (
                          <button
                            key={index}
                            onClick={(e) => {
                              e.stopPropagation();
                              action.onClick();
                            }}
                            className="flex-1 px-3 py-1.5 text-xs font-medium bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg transition-colors"
                          >
                            {action.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Recent AI Activity */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Recent AI Activity</h2>
          <button
            onClick={() => router.push('/ai/unified')}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            View All Activity
          </button>
        </div>
        
        <div className="space-y-3">
          {recentActivities.map((activity) => {
            const Icon = getActivityIcon(activity.type);
            return (
              <div
                key={activity.id}
                className="flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg cursor-pointer transition-colors"
                onClick={() => {
                  if (activity.type === 'prediction') router.push('/ai/predictive');
                  else if (activity.type === 'correlation') router.push('/ai/correlations');
                  else if (activity.type === 'chat') router.push('/ai/chat');
                  else router.push('/ai/unified');
                }}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-700`}>
                    <Icon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                  </div>
                  <div>
                    <p className="font-medium">{activity.message}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {activity.time}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {activity.status === 'success' && <CheckCircle className="h-5 w-5 text-green-500" />}
                  {activity.status === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-500" />}
                  {activity.status === 'error' && <XCircle className="h-5 w-5 text-red-500" />}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* AI Model Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Beaker className="h-5 w-5 text-purple-500" />
            Training Pipeline
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Active Jobs</span>
              <span className="font-semibold">12</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Completed Today</span>
              <span className="font-semibold text-green-600 dark:text-green-400">47</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Failed</span>
              <span className="font-semibold text-red-600 dark:text-red-400">2</span>
            </div>
            <button
              onClick={() => router.push('/ai/unified?tab=training')}
              className="w-full mt-3 px-3 py-2 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium"
            >
              Manage Training
            </button>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Cpu className="h-5 w-5 text-blue-500" />
            Model Registry
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Total Models</span>
              <span className="font-semibold">89</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Deployed</span>
              <span className="font-semibold text-green-600 dark:text-green-400">67</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">In Development</span>
              <span className="font-semibold text-yellow-600 dark:text-yellow-400">22</span>
            </div>
            <button
              onClick={() => router.push('/ai/unified?tab=models')}
              className="w-full mt-3 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium"
            >
              View Models
            </button>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-orange-500" />
            Insights Generated
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Today</span>
              <span className="font-semibold">234</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">This Week</span>
              <span className="font-semibold">1,847</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Actionable</span>
              <span className="font-semibold text-green-600 dark:text-green-400">92%</span>
            </div>
            <button
              onClick={() => router.push('/ai/correlations')}
              className="w-full mt-3 px-3 py-2 bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 rounded-lg hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors text-sm font-medium"
            >
              View Insights
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}