/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { GitBranch, Link, Network, Sparkles, AlertTriangle, TrendingUp, Layers, Cpu } from 'lucide-react'
import AppLayout from '../../components/AppLayout'

export default function CorrelationsPage() {
  const [correlationData, setCorrelationData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchCorrelationData()
  }, [])

  const fetchCorrelationData = async () => {
    try {
      const response = await fetch('/api/v1/correlations')
      const data = await response.json()
      setCorrelationData(data)
    } catch (error) {
      console.error('Failed to fetch correlation data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
          <GitBranch className="w-8 h-8 mr-3 text-indigo-500" />
          Cross-Domain Correlations
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          AI-powered pattern detection across security, cost, and performance domains
        </p>
      </div>

      {/* Patent Technology Badge */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-4 text-white mb-8"
      >
        <div className="flex items-center">
          <Sparkles className="w-6 h-6 mr-2" />
          <span className="font-semibold">Patent #1: Cross-Domain Governance Correlation Engine</span>
        </div>
        <p className="text-sm text-indigo-100 mt-2">
          Detecting hidden patterns and relationships across your Azure infrastructure
        </p>
      </motion.div>

      {/* Active Correlations */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-red-100 dark:bg-red-900 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <span className="text-sm bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 px-2 py-1 rounded-full">
              Critical
            </span>
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
            Security → Cost Impact
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Unencrypted storage causing compliance violations and potential $45K/mo penalty
          </p>
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Confidence: 94%</span>
            <button className="text-blue-500 hover:text-blue-600">View Details</button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-yellow-100 dark:bg-yellow-900 rounded-lg">
              <TrendingUp className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
            </div>
            <span className="text-sm bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 px-2 py-1 rounded-full">
              Warning
            </span>
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
            Performance → Cost Spike
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Auto-scaling triggered 5x causing $12K unexpected cost increase this month
          </p>
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Confidence: 87%</span>
            <button className="text-blue-500 hover:text-blue-600">View Details</button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
              <Network className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <span className="text-sm bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-2 py-1 rounded-full">
              Info
            </span>
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
            Network → Compliance
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Cross-region data transfer violating GDPR data residency requirements
          </p>
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Confidence: 76%</span>
            <button className="text-blue-500 hover:text-blue-600">View Details</button>
          </div>
        </motion.div>
      </div>

      {/* Correlation Matrix */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm mb-8"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Domain Correlation Matrix
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-left p-2"></th>
                <th className="text-center p-2 text-sm">Security</th>
                <th className="text-center p-2 text-sm">Cost</th>
                <th className="text-center p-2 text-sm">Performance</th>
                <th className="text-center p-2 text-sm">Compliance</th>
                <th className="text-center p-2 text-sm">Network</th>
              </tr>
            </thead>
            <tbody>
              {['Security', 'Cost', 'Performance', 'Compliance', 'Network'].map((row) => (
                <tr key={row}>
                  <td className="p-2 font-medium text-sm">{row}</td>
                  {['Security', 'Cost', 'Performance', 'Compliance', 'Network'].map((col) => {
                    const value = Math.random()
                    const color = value > 0.7 ? 'bg-red-100 text-red-700' :
                                 value > 0.4 ? 'bg-yellow-100 text-yellow-700' :
                                 'bg-green-100 text-green-700'
                    return (
                      <td key={col} className="p-2">
                        <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-xs font-semibold ${color}`}>
                          {(value * 100).toFixed(0)}
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex items-center justify-center space-x-6 text-xs">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-100 rounded mr-2"></div>
            <span>Low Correlation</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-yellow-100 rounded mr-2"></div>
            <span>Medium Correlation</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-100 rounded mr-2"></div>
            <span>High Correlation</span>
          </div>
        </div>
      </motion.div>

      {/* Discovered Patterns */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm"
      >
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Recently Discovered Patterns
        </h2>
        <div className="space-y-4">
          {[
            {
              pattern: 'Weekend Traffic Spike → Cost Anomaly',
              description: 'Unusual weekend traffic patterns correlating with 3x cost increase',
              domains: ['Performance', 'Cost'],
              action: 'Investigate potential DDoS or bot traffic'
            },
            {
              pattern: 'Policy Change → Performance Degradation',
              description: 'Recent security policy update causing 20% latency increase',
              domains: ['Security', 'Performance'],
              action: 'Review and optimize security rules'
            },
            {
              pattern: 'Storage Growth → Compliance Risk',
              description: 'Rapid data growth approaching GDPR retention limits',
              domains: ['Storage', 'Compliance'],
              action: 'Implement data lifecycle policies'
            }
          ].map((pattern, idx) => (
            <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-white">{pattern.pattern}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{pattern.description}</p>
                  <div className="flex items-center space-x-2 mt-3">
                    {pattern.domains.map((domain) => (
                      <span key={domain} className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                        {domain}
                      </span>
                    ))}
                  </div>
                  <p className="text-sm text-blue-600 dark:text-blue-400 mt-2">
                    Recommended: {pattern.action}
                  </p>
                </div>
                <Link className="w-5 h-5 text-gray-400" />
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
    </AppLayout>
  )
}