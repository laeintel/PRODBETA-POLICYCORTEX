'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Zap,
  Settings,
  UserMinus,
  Network,
  FileEdit,
  Shield,
  Trash2,
  Play,
  RotateCcw,
  Save,
  AlertTriangle,
  CheckCircle,
  Info
} from 'lucide-react'

interface WhatIfSimulatorProps {
  correlations: any[]
}

export default function WhatIfSimulator({ correlations }: WhatIfSimulatorProps) {
  const [selectedChanges, setSelectedChanges] = useState<string[]>([])
  const [simulationResult, setSimulationResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const changeTypes = [
    { id: 'role_removal', label: 'Remove Admin Role', icon: UserMinus, risk: 'high' },
    { id: 'network_segment', label: 'Add Network Segmentation', icon: Network, risk: 'low' },
    { id: 'policy_modify', label: 'Modify Security Policy', icon: FileEdit, risk: 'medium' },
    { id: 'security_control', label: 'Add Encryption', icon: Shield, risk: 'low' },
    { id: 'resource_delete', label: 'Delete Resource', icon: Trash2, risk: 'high' },
    { id: 'permission_revoke', label: 'Revoke Permissions', icon: Settings, risk: 'medium' }
  ]

  const runSimulation = async () => {
    if (selectedChanges.length === 0) return
    
    setLoading(true)
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Mock simulation result
      setSimulationResult(getMockSimulationResult(selectedChanges))
    } finally {
      setLoading(false)
    }
  }

  const toggleChange = (changeId: string) => {
    setSelectedChanges(prev => 
      prev.includes(changeId) 
        ? prev.filter(id => id !== changeId)
        : [...prev, changeId]
    )
    setSimulationResult(null)
  }

  const riskColors: Record<string, string> = {
    high: 'border-red-500/50 bg-red-500/10',
    medium: 'border-yellow-500/50 bg-yellow-500/10',
    low: 'border-green-500/50 bg-green-500/10'
  }

  return (
    <div className="space-y-6">
      {/* Change Selection */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5 text-blue-400" />
          Proposed Changes
        </h3>
        
        <div className="grid grid-cols-2 gap-4 mb-6">
          {changeTypes.map(change => {
            const Icon = change.icon
            const isSelected = selectedChanges.includes(change.id)
            
            return (
              <motion.button
                key={change.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => toggleChange(change.id)}
                className={`
                  p-4 rounded-lg border-2 transition-all text-left
                  ${isSelected 
                    ? `${riskColors[change.risk]} border-opacity-100` 
                    : 'bg-white/5 border-white/10 hover:border-white/20'
                  }
                `}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg ${
                    isSelected ? 'bg-white/20' : 'bg-white/10'
                  }`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-white mb-1">
                      {change.label}
                    </div>
                    <div className="text-xs text-gray-400">
                      Risk Level: <span className={`
                        ${change.risk === 'high' ? 'text-red-400' : 
                          change.risk === 'medium' ? 'text-yellow-400' : 'text-green-400'}
                      `}>{change.risk}</span>
                    </div>
                  </div>
                  {isSelected && (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  )}
                </div>
              </motion.button>
            )
          })}
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={runSimulation}
            disabled={selectedChanges.length === 0 || loading}
            className="px-6 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 text-white font-medium transition-all"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Simulating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Simulation
              </>
            )}
          </button>
          
          <button
            onClick={() => {
              setSelectedChanges([])
              setSimulationResult(null)
            }}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg flex items-center gap-2 text-white transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          
          {simulationResult && (
            <button className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg flex items-center gap-2 text-white transition-all">
              <Save className="w-4 h-4" />
              Save Scenario
            </button>
          )}
        </div>
      </div>

      {/* Simulation Results */}
      {simulationResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Simulation Results
          </h3>
          
          {/* Impact Summary */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <ImpactCard
              label="Affected Resources"
              value={simulationResult.affected_resources}
              change={simulationResult.resource_change}
              icon={Shield}
            />
            <ImpactCard
              label="Risk Score Delta"
              value={`${simulationResult.risk_delta > 0 ? '+' : ''}${simulationResult.risk_delta}%`}
              change={simulationResult.risk_delta}
              icon={AlertTriangle}
            />
            <ImpactCard
              label="Compliance Impact"
              value={simulationResult.compliance_impact}
              change={simulationResult.compliance_change}
              icon={FileEdit}
            />
            <ImpactCard
              label="Est. Cost Impact"
              value={simulationResult.cost_impact}
              change={simulationResult.cost_change}
              icon={Settings}
            />
          </div>
          
          {/* Detailed Analysis */}
          <div className="space-y-4">
            <div className="border-t border-white/10 pt-4">
              <h4 className="text-sm font-semibold text-gray-400 mb-3">Connectivity Changes</h4>
              <div className="grid grid-cols-3 gap-3">
                <MetricChange label="Average Degree" before="4.2" after="3.8" />
                <MetricChange label="Clustering Coefficient" before="0.68" after="0.52" />
                <MetricChange label="Network Components" before="3" after="5" />
              </div>
            </div>
            
            <div className="border-t border-white/10 pt-4">
              <h4 className="text-sm font-semibold text-gray-400 mb-3">Recommended Actions</h4>
              <div className="space-y-2">
                {simulationResult.recommendations.map((rec: string, idx: number) => (
                  <div key={idx} className="flex items-start gap-2 p-2 bg-black/20 rounded-lg">
                    <Info className="w-4 h-4 text-blue-400 mt-0.5" />
                    <span className="text-sm text-white">{rec}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="border-t border-white/10 pt-4">
              <h4 className="text-sm font-semibold text-gray-400 mb-3">Confidence Analysis</h4>
              <div className="space-y-2">
                <ConfidenceBar label="Overall Confidence" value={simulationResult.confidence.overall} />
                <ConfidenceBar label="Risk Prediction" value={simulationResult.confidence.risk} />
                <ConfidenceBar label="Cascading Effects" value={simulationResult.confidence.cascading} />
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

function ImpactCard({ label, value, change, icon: Icon }: any) {
  const isPositive = typeof change === 'number' ? change < 0 : change === 'improved'
  const isNegative = typeof change === 'number' ? change > 0 : change === 'degraded'
  
  return (
    <div className="bg-black/20 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-gray-400" />
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className={`text-lg font-semibold ${
        isPositive ? 'text-green-400' : isNegative ? 'text-red-400' : 'text-white'
      }`}>
        {value}
      </div>
    </div>
  )
}

function MetricChange({ label, before, after }: any) {
  return (
    <div className="bg-black/20 rounded-lg p-2">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="flex items-center gap-2 text-sm">
        <span className="text-gray-500">{before}</span>
        <span className="text-gray-600">→</span>
        <span className="text-white font-medium">{after}</span>
      </div>
    </div>
  )
}

function ConfidenceBar({ label, value }: any) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-400">{label}</span>
        <span className="text-xs text-white">{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="w-full h-2 bg-black/30 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`h-full bg-gradient-to-r ${
            value > 0.8 ? 'from-green-500 to-emerald-500' :
            value > 0.6 ? 'from-yellow-500 to-amber-500' :
            'from-red-500 to-orange-500'
          }`}
        />
      </div>
    </div>
  )
}

function getMockSimulationResult(changes: string[]) {
  return {
    affected_resources: 156,
    resource_change: 23,
    risk_delta: -12,
    compliance_impact: 'Improved',
    compliance_change: 'improved',
    cost_impact: '$2,400/mo',
    cost_change: 2400,
    recommendations: [
      'Review and mitigate increased risks for 23 resources',
      'Network segmentation has created isolated components - verify intended',
      'Address NIST compliance degradation (-5.2%)',
      'Consider implementing compensating controls for removed permissions'
    ],
    confidence: {
      overall: 0.85,
      risk: 0.78,
      cascading: 0.92
    },
    execution_time_ms: 420
  }
}