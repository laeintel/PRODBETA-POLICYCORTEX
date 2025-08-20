'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  TrendingUp,
  AlertTriangle,
  Shield,
  ChevronRight,
  Info,
  AlertCircle,
  Activity,
  Target,
  Zap
} from 'lucide-react'

interface RiskPropagationProps {
  correlations: any[]
  selectedNode: string | null
}

export default function RiskPropagation({ correlations, selectedNode }: RiskPropagationProps) {
  const [propagationPaths, setPropagationPaths] = useState<any[]>([])
  const [blastRadius, setBlastRadius] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (selectedNode) {
      calculateRiskPropagation(selectedNode)
    }
  }, [selectedNode])

  const calculateRiskPropagation = async (nodeId: string) => {
    setLoading(true)
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Mock propagation data
      setPropagationPaths(getMockPropagationPaths(nodeId))
      setBlastRadius(getMockBlastRadius(nodeId))
    } finally {
      setLoading(false)
    }
  }

  const amplificationColors = {
    high: 'from-red-500 to-orange-500',
    medium: 'from-yellow-500 to-amber-500',
    low: 'from-green-500 to-emerald-500'
  }

  return (
    <div className="space-y-6">
      {/* Domain Amplification Matrix */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-400" />
          Domain Amplification Matrix
        </h3>
        
        <div className="grid grid-cols-3 gap-4">
          <AmplificationCard
            domains={['Security', 'Compliance']}
            factor={1.5}
            percentage="50%"
            color="high"
          />
          <AmplificationCard
            domains={['Identity', 'Security']}
            factor={1.8}
            percentage="80%"
            color="high"
          />
          <AmplificationCard
            domains={['Network', 'Data']}
            factor={1.6}
            percentage="60%"
            color="medium"
          />
        </div>
      </div>

      {/* Blast Radius Analysis */}
      {selectedNode ? (
        <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-red-400" />
            Blast Radius Analysis
          </h3>
          
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : blastRadius ? (
            <div className="space-y-4">
              <div className="grid grid-cols-4 gap-4">
                <MetricCard
                  label="Affected Resources"
                  value={blastRadius.affected_nodes}
                  icon={Shield}
                />
                <MetricCard
                  label="Blast Radius Score"
                  value={`${(blastRadius.score * 100).toFixed(1)}%`}
                  icon={AlertTriangle}
                />
                <MetricCard
                  label="Max Propagation"
                  value={`${blastRadius.max_distance} hops`}
                  icon={TrendingUp}
                />
                <MetricCard
                  label="Computation Time"
                  value={`${blastRadius.computation_time}ms`}
                  icon={Zap}
                />
              </div>
              
              <div className="border-t border-white/10 pt-4">
                <h4 className="text-sm font-semibold text-gray-400 mb-3">Risk Propagation Paths</h4>
                <div className="space-y-2">
                  {propagationPaths.map((path, idx) => (
                    <PropagationPath key={idx} path={path} />
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Select a node to analyze blast radius</p>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-12">
          <div className="text-center text-gray-400">
            <Info className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>Select a node from the correlation graph to analyze risk propagation</p>
          </div>
        </div>
      )}

      {/* Risk Cascade Visualization */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-yellow-400" />
          Active Risk Cascades
        </h3>
        
        <div className="space-y-3">
          {correlations.slice(0, 5).map((corr, idx) => (
            <RiskCascadeItem key={idx} correlation={corr} index={idx} />
          ))}
        </div>
      </div>
    </div>
  )
}

function AmplificationCard({ domains, factor, percentage, color }: any) {
  const colorClasses: Record<string, string> = {
    high: 'from-red-500 to-orange-500',
    medium: 'from-yellow-500 to-amber-500',
    low: 'from-green-500 to-emerald-500'
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-white/5 border border-white/10 rounded-lg p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className={`w-2 h-8 bg-gradient-to-b ${colorClasses[color]} rounded-full`} />
        <span className="text-2xl font-bold text-white">{factor}x</span>
      </div>
      <div className="space-y-1">
        <div className="text-sm font-medium text-white">
          {domains[0]} + {domains[1]}
        </div>
        <div className="text-xs text-gray-400">
          {percentage} risk increase
        </div>
      </div>
    </motion.div>
  )
}

function MetricCard({ label, value, icon: Icon }: any) {
  return (
    <div className="bg-black/20 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-4 h-4 text-gray-400" />
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className="text-lg font-semibold text-white">{value}</div>
    </div>
  )
}

function PropagationPath({ path }: any) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-2 p-3 bg-black/20 rounded-lg"
    >
      <div className="flex-1 flex items-center gap-2">
        {path.nodes.map((node: string, idx: number) => (
          <div key={idx} className="flex items-center gap-2">
            <div className="px-2 py-1 bg-white/10 rounded text-xs text-white">
              {node}
            </div>
            {idx < path.nodes.length - 1 && (
              <ChevronRight className="w-3 h-3 text-gray-500" />
            )}
          </div>
        ))}
      </div>
      <div className="text-sm">
        <span className="text-gray-400">Risk:</span>
        <span className={`ml-1 font-semibold ${
          path.total_risk > 0.7 ? 'text-red-400' : 
          path.total_risk > 0.4 ? 'text-yellow-400' : 'text-green-400'
        }`}>
          {(path.total_risk * 100).toFixed(0)}%
        </span>
      </div>
    </motion.div>
  )
}

function RiskCascadeItem({ correlation, index }: any) {
  const riskLevel = correlation.risk_amplification > 1.6 ? 'high' : 
                    correlation.risk_amplification > 1.3 ? 'medium' : 'low'
  
  const riskColors = {
    high: 'bg-red-500',
    medium: 'bg-yellow-500',
    low: 'bg-green-500'
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className="flex items-center gap-4 p-3 bg-black/20 rounded-lg hover:bg-black/30 transition-colors"
      >
        <div className={`w-1 h-full ${riskColors[riskLevel]} rounded-full`} />
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-white">
              {correlation.source.id} → {correlation.target.id}
            </span>
            <span className="text-xs px-2 py-0.5 bg-white/10 rounded text-gray-400">
              {correlation.domain_pair.join(' / ')}
            </span>
          </div>
          <div className="text-xs text-gray-400">
            {correlation.description}
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm font-semibold text-white">
            {correlation.risk_amplification}x
          </div>
          <div className="text-xs text-gray-400">
            amplification
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

function getMockPropagationPaths(nodeId: string) {
  return [
    {
      nodes: [nodeId, 'db-main', 'api-gateway', 'web-frontend'],
      total_risk: 0.85,
      decay_factor: 0.9
    },
    {
      nodes: [nodeId, 'iam-service', 'auth-provider'],
      total_risk: 0.72,
      decay_factor: 0.85
    },
    {
      nodes: [nodeId, 'network-dmz', 'load-balancer', 'cdn'],
      total_risk: 0.56,
      decay_factor: 0.8
    }
  ]
}

function getMockBlastRadius(nodeId: string) {
  return {
    affected_nodes: 47,
    score: 0.73,
    max_distance: 4,
    computation_time: 92,
    critical_resources: ['db-main', 'auth-service', 'payment-gateway']
  }
}