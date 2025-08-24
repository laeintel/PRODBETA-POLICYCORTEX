'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  Cpu,
  Zap,
  Target,
  TrendingUp,
  TrendingDown,
  Activity,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Clock,
  Settings,
  RefreshCw,
  Eye,
  Play,
  Pause,
  BarChart3,
  LineChart,
  PieChart,
  Sparkles,
  Bot,
  MessageSquare,
  Search,
  Filter,
  Database,
  Cloud,
  Network,
  Shield,
  Users,
  FileText,
  Lightbulb,
  Gauge,
  Globe
} from 'lucide-react'
import ViewToggle from '@/components/ViewToggle';
import ChartContainer from '@/components/ChartContainer';
import MetricCard from '@/components/MetricCard';

export default function TacticalAIPage() {
  const [view, setView] = useState<'cards' | 'visualizations'>('cards');
  const [aiMetrics, setAiMetrics] = useState({
    modelAccuracy: 94.7,
    inferenceLatency: 23,
    predictionConfidence: 87.3,
    aiRecommendations: 156,
    automatedActions: 89
  })

  const [aiModels, setAiModels] = useState([
    {
      name: 'Policy Compliance Predictor',
      type: 'Classification',
      status: 'active',
      accuracy: 96.8,
      latency: 15,
      requests: 15420,
      version: '2.1.0',
      lastTrained: new Date(Date.now() - 432000000),
      features: ['Policy Violations', 'Resource Configurations', 'Historical Patterns']
    },
    {
      name: 'Cost Optimization Engine',
      type: 'Regression',
      status: 'active',
      accuracy: 93.4,
      latency: 32,
      requests: 8750,
      version: '1.8.2',
      lastTrained: new Date(Date.now() - 259200000),
      features: ['Resource Utilization', 'Pricing Trends', 'Usage Patterns']
    },
    {
      name: 'Security Threat Detector',
      type: 'Anomaly Detection',
      status: 'active',
      accuracy: 91.2,
      latency: 18,
      requests: 23460,
      version: '3.0.1',
      lastTrained: new Date(Date.now() - 172800000),
      features: ['Network Traffic', 'Access Patterns', 'Behavioral Analysis']
    },
    {
      name: 'Resource Rightsizing Advisor',
      type: 'Recommendation',
      status: 'training',
      accuracy: 88.9,
      latency: 45,
      requests: 4230,
      version: '1.5.0-beta',
      lastTrained: new Date(Date.now() - 86400000),
      features: ['Performance Metrics', 'Cost Data', 'Usage History']
    },
    {
      name: 'Compliance Risk Assessor',
      type: 'Risk Analysis',
      status: 'active',
      accuracy: 92.1,
      latency: 28,
      requests: 9870,
      version: '2.3.1',
      lastTrained: new Date(Date.now() - 345600000),
      features: ['Regulatory Rules', 'Policy Frameworks', 'Audit History']
    },
    {
      name: 'Conversational AI Assistant',
      type: 'NLP',
      status: 'active',
      accuracy: 89.7,
      latency: 67,
      requests: 34560,
      version: '4.2.0',
      lastTrained: new Date(Date.now() - 518400000),
      features: ['Intent Recognition', 'Entity Extraction', 'Context Understanding']
    }
  ])

  const [aiRecommendations, setAiRecommendations] = useState([
    {
      id: 'rec-001',
      type: 'cost_optimization',
      title: 'Reduce VM Sizes in Development Environment',
      description: 'AI detected 67% average CPU utilization in dev VMs. Recommend downsizing to save $2,400/month.',
      confidence: 94.2,
      impact: 'high',
      estimatedSavings: 2400,
      status: 'pending',
      createdAt: new Date(Date.now() - 1800000),
      model: 'Cost Optimization Engine'
    },
    {
      id: 'rec-002',
      type: 'security',
      title: 'Update Firewall Rules for Suspicious Traffic',
      description: 'Anomalous network patterns detected from IP range 192.168.1.0/24. Recommend restricting access.',
      confidence: 87.8,
      impact: 'critical',
      estimatedSavings: 0,
      status: 'approved',
      createdAt: new Date(Date.now() - 3600000),
      model: 'Security Threat Detector'
    },
    {
      id: 'rec-003',
      type: 'compliance',
      title: 'Schedule GDPR Compliance Review',
      description: 'AI predicts potential compliance drift in data handling processes. Recommend immediate review.',
      confidence: 91.3,
      impact: 'medium',
      estimatedSavings: 0,
      status: 'implemented',
      createdAt: new Date(Date.now() - 7200000),
      model: 'Compliance Risk Assessor'
    },
    {
      id: 'rec-004',
      type: 'performance',
      title: 'Scale Database Read Replicas',
      description: 'Query response times increasing. AI recommends adding 2 read replicas to handle growing load.',
      confidence: 89.6,
      impact: 'medium',
      estimatedSavings: -800,
      status: 'pending',
      createdAt: new Date(Date.now() - 10800000),
      model: 'Resource Rightsizing Advisor'
    }
  ])

  const [automatedActions, setAutomatedActions] = useState([
    {
      id: 'action-001',
      type: 'auto_scaling',
      title: 'Auto-scaled Web Tier',
      description: 'Added 2 instances to handle traffic spike detected by predictive model',
      timestamp: new Date(Date.now() - 900000),
      status: 'completed',
      savings: -120,
      model: 'Resource Rightsizing Advisor'
    },
    {
      id: 'action-002',
      type: 'security_block',
      title: 'Blocked Suspicious IP',
      description: 'Automatically blocked IP 45.123.78.90 due to anomalous access patterns',
      timestamp: new Date(Date.now() - 2700000),
      status: 'completed',
      savings: 0,
      model: 'Security Threat Detector'
    },
    {
      id: 'action-003',
      type: 'cost_optimization',
      title: 'Stopped Idle Resources',
      description: 'Shut down 5 idle development VMs to reduce costs',
      timestamp: new Date(Date.now() - 5400000),
      status: 'completed',
      savings: 340,
      model: 'Cost Optimization Engine'
    }
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400'
      case 'training': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400'
      case 'inactive': return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
      case 'error': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-400'
      case 'high': return 'text-orange-600 bg-orange-100 dark:bg-orange-900/30 dark:text-orange-400'
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'low': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-400'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'cost_optimization': return <TrendingDown className="w-4 h-4 text-green-500" />
      case 'security': return <Shield className="w-4 h-4 text-red-500" />
      case 'compliance': return <FileText className="w-4 h-4 text-blue-500" />
      case 'performance': return <Zap className="w-4 h-4 text-purple-500" />
      case 'auto_scaling': return <TrendingUp className="w-4 h-4 text-blue-500" />
      case 'security_block': return <Shield className="w-4 h-4 text-red-500" />
      default: return <Bot className="w-4 h-4 text-gray-500" />
    }
  }

  const getModelTypeIcon = (type: string) => {
    switch (type) {
      case 'Classification': return <Target className="w-4 h-4" />
      case 'Regression': return <TrendingUp className="w-4 h-4" />
      case 'Anomaly Detection': return <Search className="w-4 h-4" />
      case 'Recommendation': return <Lightbulb className="w-4 h-4" />
      case 'Risk Analysis': return <Shield className="w-4 h-4" />
      case 'NLP': return <MessageSquare className="w-4 h-4" />
      default: return <Brain className="w-4 h-4" />
    }
  }

  const metrics = [
    {
      id: 'model-accuracy',
      title: 'Model Accuracy',
      value: `${aiMetrics.modelAccuracy}%`,
      change: 1.8,
      trend: 'up' as const,
      sparklineData: [92.1, 92.8, 93.4, 93.9, 94.3, aiMetrics.modelAccuracy],
      alert: `${aiModels.filter(m => m.status === 'training').length} models training`
    },
    {
      id: 'inference-latency',
      title: 'Avg Inference Latency',
      value: `${aiMetrics.inferenceLatency}ms`,
      change: -8.2,
      trend: 'down' as const,
      sparklineData: [28, 26, 25, 24, 23.5, aiMetrics.inferenceLatency]
    },
    {
      id: 'ai-recommendations',
      title: 'AI Recommendations',
      value: aiMetrics.aiRecommendations,
      change: 12.4,
      trend: 'up' as const,
      sparklineData: [134, 142, 148, 152, 154, aiMetrics.aiRecommendations]
    },
    {
      id: 'automated-actions',
      title: 'Automated Actions (24h)',
      value: aiMetrics.automatedActions,
      change: 23.6,
      trend: 'up' as const,
      sparklineData: [68, 72, 76, 82, 85, aiMetrics.automatedActions]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
              <Brain className="h-10 w-10 text-purple-600" />
              Tactical AI Command Center
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              AI-powered insights, predictions, and automated governance actions
            </p>
          </div>
          <div className="flex gap-3">
            <ViewToggle view={view} onViewChange={setView} />
            <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
              <RefreshCw className="h-5 w-5" />
            </button>
            <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* AI Status Alert */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 rounded-xl"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-full">
              <Sparkles className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100">
                AI Status: {aiModels.filter(m => m.status === 'active').length} Models Active
              </h3>
              <p className="text-purple-700 dark:text-purple-300">
                {aiMetrics.aiRecommendations} recommendations generated with {aiMetrics.predictionConfidence}% average confidence. 
                {aiMetrics.automatedActions} automated actions executed in the last 24 hours.
              </p>
            </div>
            <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
              View AI Dashboard
            </button>
          </div>
        </motion.div>

        {/* AI Metrics */}
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
            {/* AI Models Status */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 mb-8">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <Cpu className="w-6 h-6 text-purple-500" />
                AI Models Status
              </h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {aiModels.map((model) => (
                  <motion.div
                    key={model.name}
                    whileHover={{ scale: 1.01 }}
                    className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                          {getModelTypeIcon(model.type)}
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {model.name}
                          </h3>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-gray-500">{model.type}</span>
                            <span className="text-xs text-gray-500">v{model.version}</span>
                          </div>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(model.status)}`}>
                        {model.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-sm mb-3">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Accuracy</span>
                        <div className="font-semibold text-green-600 dark:text-green-400">
                          {model.accuracy}%
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Latency</span>
                        <div className="font-semibold text-blue-600 dark:text-blue-400">
                          {model.latency}ms
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Requests</span>
                        <div className="font-semibold text-gray-900 dark:text-white">
                          {model.requests.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">
                      Last trained: {model.lastTrained.toLocaleDateString()}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* AI Recommendations and Automated Actions */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Lightbulb className="w-6 h-6 text-yellow-500" />
                  AI Recommendations
                </h2>
                <div className="space-y-4">
                  {aiRecommendations.map((rec) => (
                    <motion.div
                      key={rec.id}
                      whileHover={{ scale: 1.01 }}
                      className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-3">
                          {getTypeIcon(rec.type)}
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">
                              {rec.title}
                            </h3>
                            <div className="flex items-center gap-2 mt-1">
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getImpactColor(rec.impact)}`}>
                                {rec.impact.toUpperCase()}
                              </span>
                              <span className="text-xs text-blue-600 dark:text-blue-400">
                                {rec.confidence}% confidence
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          {rec.estimatedSavings > 0 && (
                            <div className="text-sm font-semibold text-green-600 dark:text-green-400">
                              ${rec.estimatedSavings}/mo
                            </div>
                          )}
                          <div className="text-xs text-gray-500">
                            {rec.createdAt.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {rec.description}
                      </p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>Model: {rec.model}</span>
                        <div className="flex gap-2">
                          <button className="px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700">
                            Approve
                          </button>
                          <button className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-xs hover:bg-gray-200 dark:hover:bg-gray-600">
                            Dismiss
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Bot className="w-6 h-6 text-green-500" />
                  Automated Actions
                </h2>
                <div className="space-y-4">
                  {automatedActions.map((action) => (
                    <motion.div
                      key={action.id}
                      whileHover={{ scale: 1.01 }}
                      className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-3">
                          {getTypeIcon(action.type)}
                          <div>
                            <h3 className="font-semibold text-gray-900 dark:text-white">
                              {action.title}
                            </h3>
                            <span className="text-xs text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400 px-2 py-1 rounded-full">
                              {action.status.toUpperCase()}
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          {action.savings !== 0 && (
                            <div className={`text-sm font-semibold ${
                              action.savings > 0 
                                ? 'text-green-600 dark:text-green-400' 
                                : 'text-red-600 dark:text-red-400'
                            }`}>
                              {action.savings > 0 ? '+' : ''}${action.savings}
                            </div>
                          )}
                          <div className="text-xs text-gray-500">
                            {action.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {action.description}
                      </p>
                      <div className="text-xs text-gray-500">
                        Model: {action.model}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>

            {/* AI Capabilities Overview */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
                AI Capabilities & Features
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { name: 'Predictive Analytics', icon: TrendingUp, color: 'blue', description: 'Forecast future trends and risks' },
                  { name: 'Anomaly Detection', icon: Search, color: 'red', description: 'Identify unusual patterns and behaviors' },
                  { name: 'Automated Remediation', icon: Bot, color: 'green', description: 'Self-healing system responses' },
                  { name: 'Natural Language Processing', icon: MessageSquare, color: 'purple', description: 'Conversational AI interface' }
                ].map((capability) => {
                  const Icon = capability.icon
                  return (
                    <motion.div
                      key={capability.name}
                      whileHover={{ scale: 1.02 }}
                      className={`p-4 rounded-lg border-2 border-dashed border-${capability.color}-300 hover:border-${capability.color}-500 hover:bg-${capability.color}-50 dark:hover:bg-${capability.color}-900/20 transition-all`}
                    >
                      <Icon className={`w-6 h-6 text-${capability.color}-600 dark:text-${capability.color}-400 mx-auto mb-2`} />
                      <div className="text-sm font-medium text-gray-900 dark:text-white text-center mb-2">
                        {capability.name}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 text-center">
                        {capability.description}
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Visualization Mode */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <ChartContainer
                title="AI Model Performance Trends"
                onDrillIn={() => console.log('Drill into model performance')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">AI model performance timeline visualization</p>
                  </div>
                </div>
              </ChartContainer>
              <ChartContainer
                title="Recommendation Impact Analysis"
                onDrillIn={() => console.log('Drill into recommendation impact')}
              >
                <div className="p-4">
                  <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded">
                    <p className="text-gray-500">Recommendation impact scatter plot</p>
                  </div>
                </div>
              </ChartContainer>
            </div>
            
            {/* AI Performance Dashboard */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Gauge className="h-6 w-6 text-purple-600" />
                AI Performance Dashboard
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-purple-600">{aiMetrics.modelAccuracy}%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Model Accuracy</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-blue-600">{aiMetrics.inferenceLatency}ms</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Inference Latency</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-green-600">{aiMetrics.aiRecommendations}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Recommendations</div>
                </div>
                <div className="p-4 border dark:border-gray-700 rounded-lg text-center">
                  <div className="text-3xl font-bold text-orange-600">{aiMetrics.automatedActions}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Automated Actions</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}