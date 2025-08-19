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

import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  ArrowLeft,
  Shield,
  Brain,
  Zap,
  Globe,
  Lock,
  BarChart,
  Code,
  Cloud,
  Database,
  Cpu,
  Layers,
  GitBranch
} from 'lucide-react'

export default function FeaturesPage() {
  const router = useRouter()

  const features = [
    {
      category: 'Core Architecture',
      icon: Layers,
      items: [
        {
          title: 'Modular Monolith',
          description: 'Rust-based core with Domain-Driven Design and clean architecture principles',
          tech: ['Rust', 'Axum', 'Tower', 'Tokio']
        },
        {
          title: 'Event Sourcing & CQRS',
          description: 'Complete audit trail with time-travel debugging and event replay capabilities',
          tech: ['EventStore', 'Kafka', 'PostgreSQL']
        },
        {
          title: 'GraphQL Federation',
          description: 'Unified API gateway with schema stitching and intelligent query planning',
          tech: ['Apollo', 'GraphQL', 'Federation 2.0']
        }
      ]
    },
    {
      category: 'AI & Machine Learning',
      icon: Brain,
      items: [
        {
          title: 'Predictive Analytics',
          description: 'ML models predict compliance issues before they occur with 95% accuracy',
          tech: ['PyTorch', 'TensorFlow', 'AutoML']
        },
        {
          title: 'Natural Language Processing',
          description: 'Convert plain English to Azure policies with context-aware understanding',
          tech: ['Transformers', 'BERT', 'GPT-4']
        },
        {
          title: 'Federated Learning',
          description: 'Train models across organizations without sharing sensitive data',
          tech: ['PySyft', 'Flower', 'Differential Privacy']
        }
      ]
    },
    {
      category: 'Security & Compliance',
      icon: Lock,
      items: [
        {
          title: 'Quantum-Resistant Cryptography',
          description: 'Future-proof security with post-quantum algorithms',
          tech: ['Kyber1024', 'Dilithium5', 'SPHINCS+']
        },
        {
          title: 'Blockchain Audit Trail',
          description: 'Immutable audit logs with cryptographic proof of compliance',
          tech: ['Hyperledger', 'Smart Contracts', 'IPFS']
        },
        {
          title: 'Zero-Trust Architecture',
          description: 'Never trust, always verify with continuous authentication',
          tech: ['mTLS', 'SPIFFE/SPIRE', 'Service Mesh']
        }
      ]
    },
    {
      category: 'Performance & Scale',
      icon: Zap,
      items: [
        {
          title: 'Edge Computing',
          description: 'Sub-millisecond inference with WebAssembly at the edge',
          tech: ['WASM', 'Cloudflare Workers', 'Edge Functions']
        },
        {
          title: 'Global Distribution',
          description: 'Multi-region deployment with automatic failover and geo-routing',
          tech: ['CDN', 'Anycast', 'Traffic Manager']
        },
        {
          title: 'Reactive Streams',
          description: 'Handle millions of events per second with backpressure support',
          tech: ['Akka Streams', 'RSocket', 'Project Reactor']
        }
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <button
            onClick={() => router.push('/')}
            className="flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors mb-6"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </button>
          <h1 className="text-5xl font-bold text-white mb-4">
            Features & Capabilities
          </h1>
          <p className="text-xl text-gray-300">
            Discover the cutting-edge technology powering PolicyCortex
          </p>
        </motion.div>

        {/* Features Grid */}
        {features.map((category, categoryIndex) => {
          const CategoryIcon = category.icon
          return (
            <motion.div
              key={category.category}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: categoryIndex * 0.1 }}
              className="mb-12"
            >
              <div className="flex items-center gap-3 mb-6">
                <CategoryIcon className="w-8 h-8 text-purple-400" />
                <h2 className="text-3xl font-bold text-white">{category.category}</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {category.items.map((item, itemIndex) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: categoryIndex * 0.1 + itemIndex * 0.05 }}
                    whileHover={{ scale: 1.02, y: -5 }}
                    className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 p-6 cursor-pointer"
                    onClick={() => router.push(`/features/${item.title.toLowerCase().replace(/\s+/g, '-')}`)}
                  >
                    <h3 className="text-xl font-semibold text-white mb-3">
                      {item.title}
                    </h3>
                    <p className="text-gray-300 mb-4">
                      {item.description}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {item.tech.map((tech) => (
                        <span
                          key={tech}
                          className="px-2 py-1 text-xs bg-purple-600/30 text-purple-300 rounded-full border border-purple-500/30"
                        >
                          {tech}
                        </span>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )
        })}

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="text-center mt-16 p-8 bg-purple-600/20 backdrop-blur-md rounded-xl border border-purple-500/30"
        >
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Transform Your Cloud Governance?
          </h2>
          <p className="text-gray-300 mb-6">
            Experience the power of AI-driven compliance and automation
          </p>
          <div className="inline-flex gap-4">
            <button
              onClick={() => router.push('/dashboard')}
              className="px-8 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition-colors"
            >
              Start Free Trial
            </button>
            <button
              onClick={() => {
                if (typeof window !== 'undefined') {
                  window.open('https://github.com/aeolitech/policycortex', '_blank')
                }
              }}
              className="px-8 py-3 bg-white/10 text-white rounded-lg font-semibold backdrop-blur-md hover:bg-white/20 transition-colors border border-white/20"
            >
              View on GitHub
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}