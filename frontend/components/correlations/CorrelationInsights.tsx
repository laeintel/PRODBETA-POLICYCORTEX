'use client'

import { motion } from 'framer-motion'
import {
  Brain,
  TrendingUp,
  BarChart3,
  Info,
  AlertCircle
} from 'lucide-react'

interface CorrelationInsightsProps {
  correlations: any[]
}

export default function CorrelationInsights({ correlations }: CorrelationInsightsProps) {
  // SHAP feature importance mock data
  const featureImportance = [
    { feature: 'encryption_status', importance: 0.32, contribution: 'positive' },
    { feature: 'public_exposure', importance: 0.28, contribution: 'negative' },
    { feature: 'identity_permissions', importance: 0.22, contribution: 'negative' },
    { feature: 'network_segmentation', importance: 0.18, contribution: 'positive' },
    { feature: 'compliance_score', importance: 0.15, contribution: 'positive' },
    { feature: 'data_classification', importance: 0.12, contribution: 'negative' }
  ]

  const attentionWeights = [
    { source: 'IAM Policy', target: 'S3 Bucket', weight: 0.89 },
    { source: 'Network ACL', target: 'EC2 Instance', weight: 0.76 },
    { source: 'Security Group', target: 'RDS Database', weight: 0.71 },
    { source: 'KMS Key', target: 'Lambda Function', weight: 0.65 }
  ]

  return (
    <div className="space-y-6">
      {/* ML Model Performance */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          Graph Neural Network Performance
        </h3>
        
        <div className="grid grid-cols-4 gap-4">
          <PerformanceMetric label="Inference Time" value="850ms" target="<1000ms" status="success" />
          <PerformanceMetric label="Accuracy" value="96.2%" target="95%" status="success" />
          <PerformanceMetric label="Nodes Processed" value="10,234" target="100k max" status="success" />
          <PerformanceMetric label="Memory Usage" value="2.1GB" target="<4GB" status="success" />
        </div>
      </div>

      {/* SHAP Feature Importance */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-400" />
          Feature Importance (SHAP Values)
        </h3>
        
        <div className="space-y-3">
          {featureImportance.map((feature, idx) => (
            <FeatureImportanceBar key={idx} {...feature} rank={idx + 1} />
          ))}
        </div>
        
        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-blue-400 mt-0.5" />
            <div className="text-sm text-gray-300">
              <strong className="text-white">Interpretation:</strong> Encryption status and public exposure 
              are the strongest predictors of cross-domain correlations, accounting for 60% of the model's 
              decision-making process.
            </div>
          </div>
        </div>
      </div>

      {/* Attention Weights */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-green-400" />
          Attention Mechanism Weights
        </h3>
        
        <div className="space-y-3">
          {attentionWeights.map((attention, idx) => (
            <AttentionWeightItem key={idx} {...attention} />
          ))}
        </div>
        
        <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5" />
            <div className="text-sm text-gray-300">
              <strong className="text-white">High Attention Alert:</strong> The model is focusing heavily on 
              IAM-to-Storage relationships, indicating potential permission escalation risks.
            </div>
          </div>
        </div>
      </div>

      {/* Correlation Patterns */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Discovered Patterns</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <PatternCard
            title="Toxic Combination #1"
            description="Public S3 bucket with IAM admin role"
            severity="critical"
            frequency="3 instances"
          />
          <PatternCard
            title="Toxic Combination #2"
            description="Unencrypted RDS with public endpoint"
            severity="high"
            frequency="5 instances"
          />
          <PatternCard
            title="Risk Cascade Pattern"
            description="Identity → Security → Compliance chain"
            severity="medium"
            frequency="12 instances"
          />
          <PatternCard
            title="Cost Correlation"
            description="Over-provisioned resources in same VPC"
            severity="low"
            frequency="28 instances"
          />
        </div>
      </div>
    </div>
  )
}

function PerformanceMetric({ label, value, target, status }: any) {
  return (
    <div className="bg-black/20 rounded-lg p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-xl font-bold text-white mb-1">{value}</div>
      <div className="text-xs text-gray-500">Target: {target}</div>
      <div className="mt-2">
        <div className={`w-full h-1 rounded-full ${
          status === 'success' ? 'bg-green-500' : 'bg-yellow-500'
        }`} />
      </div>
    </div>
  )
}

function FeatureImportanceBar({ feature, importance, contribution, rank }: any) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-6 text-center text-sm text-gray-500">{rank}</div>
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-white">{feature.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</span>
          <span className="text-sm font-medium text-white">{(importance * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full h-2 bg-black/30 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${importance * 100}%` }}
            transition={{ duration: 0.5, delay: rank * 0.1 }}
            className={`h-full bg-gradient-to-r ${
              contribution === 'positive' 
                ? 'from-green-500 to-emerald-500' 
                : 'from-red-500 to-orange-500'
            }`}
          />
        </div>
      </div>
    </div>
  )
}

function AttentionWeightItem({ source, target, weight }: any) {
  return (
    <div className="flex items-center gap-3 p-3 bg-black/20 rounded-lg">
      <div className="flex-1 flex items-center gap-2">
        <span className="text-sm text-white">{source}</span>
        <span className="text-gray-600">→</span>
        <span className="text-sm text-white">{target}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-32 h-2 bg-black/30 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${weight * 100}%` }}
            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
          />
        </div>
        <span className="text-sm font-medium text-white w-12 text-right">
          {(weight * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

function PatternCard({ title, description, severity, frequency }: any) {
  const severityColors: Record<string, string> = {
    critical: 'border-red-500/50 bg-red-500/10',
    high: 'border-orange-500/50 bg-orange-500/10',
    medium: 'border-yellow-500/50 bg-yellow-500/10',
    low: 'border-green-500/50 bg-green-500/10'
  }

  const severityLabels: Record<string, string> = {
    critical: 'text-red-400',
    high: 'text-orange-400',
    medium: 'text-yellow-400',
    low: 'text-green-400'
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className={`p-4 rounded-lg border ${severityColors[severity]}`}
    >
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-medium text-white">{title}</h4>
        <span className={`text-xs font-medium ${severityLabels[severity]} uppercase`}>
          {severity}
        </span>
      </div>
      <p className="text-sm text-gray-400 mb-3">{description}</p>
      <div className="text-xs text-gray-500">Frequency: {frequency}</div>
    </motion.div>
  )
}