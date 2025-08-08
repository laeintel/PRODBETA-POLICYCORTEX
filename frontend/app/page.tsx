'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Shield, Brain, Zap, Globe, Lock, BarChart } from 'lucide-react'

export default function HomePage() {
  const [activeFeature, setActiveFeature] = useState(0)

  const features = [
    {
      icon: Shield,
      title: 'Policy Management',
      description: 'AI-powered policy creation and enforcement with real-time compliance monitoring'
    },
    {
      icon: Brain,
      title: 'Intelligent Insights',
      description: 'Machine learning algorithms provide predictive analytics and optimization recommendations'
    },
    {
      icon: Zap,
      title: 'Edge Computing',
      description: 'Sub-millisecond inference with WebAssembly at the edge for instant decisions'
    },
    {
      icon: Globe,
      title: 'Global Scale',
      description: 'Distributed architecture ensures high availability across all Azure regions'
    },
    {
      icon: Lock,
      title: 'Quantum-Ready Security',
      description: 'Post-quantum cryptography protects against future threats'
    },
    {
      icon: BarChart,
      title: 'Advanced Analytics',
      description: 'Comprehensive dashboards with real-time metrics and custom reports'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h1 className="text-6xl font-bold text-white mb-4">
            PolicyCortex <span className="text-purple-400">v2</span>
          </h1>
          <p className="text-xl text-gray-300">
            AI-Powered Azure Governance Platform
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16"
        >
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onHoverStart={() => setActiveFeature(index)}
                className={`p-6 rounded-xl backdrop-blur-md transition-all cursor-pointer ${
                  activeFeature === index
                    ? 'bg-purple-800/30 border-purple-400'
                    : 'bg-white/10 border-white/20'
                } border`}
              >
                <Icon className="w-12 h-12 text-purple-400 mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-300">
                  {feature.description}
                </p>
              </motion.div>
            )
          })}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="text-center"
        >
          <div className="inline-flex gap-4">
            <button className="px-8 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition-colors">
              Get Started
            </button>
            <button className="px-8 py-3 bg-white/10 text-white rounded-lg font-semibold backdrop-blur-md hover:bg-white/20 transition-colors border border-white/20">
              Learn More
            </button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.5 }}
          className="mt-16 text-center text-gray-400"
        >
          <p>🚀 Completely rebuilt with 80 architectural improvements</p>
          <p className="mt-2">
            Rust • Next.js 14 • GraphQL • WebAssembly • Blockchain • Quantum Computing
          </p>
        </motion.div>
      </div>
    </div>
  )
}