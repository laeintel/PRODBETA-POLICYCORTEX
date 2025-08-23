'use client'

import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Cloud, CheckCircle, AlertCircle, Clock, Server, Database, Shield } from 'lucide-react'

interface CloudProvider {
  name: string
  status: 'connected' | 'disconnected' | 'pending'
  resources: number
  lastSync: string
  color: string
  icon: string
  details: {
    subscriptions?: number
    regions?: string[]
    services?: string[]
  }
}

export default function CloudIntegrationStatus() {
  const [providers, setProviders] = useState<CloudProvider[]>([
    {
      name: 'Microsoft Azure',
      status: 'connected',
      resources: 247,
      lastSync: '2 minutes ago',
      color: 'from-blue-500 to-blue-600',
      icon: '☁️',
      details: {
        subscriptions: 3,
        regions: ['East US', 'West Europe', 'Southeast Asia'],
        services: ['VMs', 'Storage', 'AKS', 'Functions', 'SQL DB']
      }
    },
    {
      name: 'Amazon AWS',
      status: 'connected',
      resources: 189,
      lastSync: '5 minutes ago',
      color: 'from-orange-500 to-orange-600',
      icon: '🔶',
      details: {
        subscriptions: 2,
        regions: ['us-east-1', 'eu-west-1'],
        services: ['EC2', 'S3', 'RDS', 'Lambda', 'EKS']
      }
    },
    {
      name: 'Google Cloud',
      status: 'pending',
      resources: 0,
      lastSync: 'Never',
      color: 'from-green-500 to-green-600',
      icon: '🟢',
      details: {
        subscriptions: 0,
        regions: [],
        services: []
      }
    }
  ])

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setProviders(prev => prev.map(p => ({
        ...p,
        lastSync: p.status === 'connected' ? 'Just now' : p.lastSync
      })))
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const handleConnect = (providerName: string) => {
    console.log(`Connecting to ${providerName}...`)
    // Simulate connection process
    setProviders(prev => prev.map(p => 
      p.name === providerName 
        ? { ...p, status: 'pending' as const }
        : p
    ))
    
    setTimeout(() => {
      setProviders(prev => prev.map(p => 
        p.name === providerName 
          ? { 
              ...p, 
              status: 'connected' as const,
              resources: Math.floor(Math.random() * 100) + 50,
              lastSync: 'Just now'
            }
          : p
      ))
    }, 2000)
  }

  const handleRefresh = (providerName: string) => {
    console.log(`Refreshing ${providerName}...`)
    setProviders(prev => prev.map(p => 
      p.name === providerName 
        ? { ...p, lastSync: 'Refreshing...' }
        : p
    ))
    
    setTimeout(() => {
      setProviders(prev => prev.map(p => 
        p.name === providerName 
          ? { ...p, lastSync: 'Just now', resources: p.resources + Math.floor(Math.random() * 10) - 5 }
          : p
      ))
    }, 1000)
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Multi-Cloud Integration
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Total Resources: {providers.reduce((sum, p) => sum + p.resources, 0)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {providers.map((provider) => (
          <motion.div
            key={provider.name}
            whileHover={{ scale: 1.02 }}
            className={`relative rounded-lg p-4 border-2 transition-all ${
              provider.status === 'connected' 
                ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                : provider.status === 'pending'
                ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
                : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-900/20'
            }`}
          >
            <div className={`absolute inset-0 bg-gradient-to-br ${provider.color} opacity-10 rounded-lg`} />
            
            <div className="relative">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{provider.icon}</span>
                  <h3 className="font-semibold text-gray-900 dark:text-white">
                    {provider.name}
                  </h3>
                </div>
                {provider.status === 'connected' ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : provider.status === 'pending' ? (
                  <Clock className="w-5 h-5 text-yellow-500 animate-pulse" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-gray-400" />
                )}
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Status</span>
                  <span className={`font-medium ${
                    provider.status === 'connected' 
                      ? 'text-green-600 dark:text-green-400'
                      : provider.status === 'pending'
                      ? 'text-yellow-600 dark:text-yellow-400'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}>
                    {provider.status}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Resources</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {provider.resources}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Last Sync</span>
                  <span className="text-gray-900 dark:text-white">
                    {provider.lastSync}
                  </span>
                </div>

                {provider.status === 'connected' && provider.details.services && (
                  <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex flex-wrap gap-1">
                      {provider.details.services.slice(0, 3).map(service => (
                        <span key={service} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-700 dark:text-gray-300">
                          {service}
                        </span>
                      ))}
                      {provider.details.services.length > 3 && (
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-700 dark:text-gray-300">
                          +{provider.details.services.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-4 flex gap-2">
                {provider.status === 'disconnected' ? (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleConnect(provider.name)}
                    className="flex-1 px-3 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white text-sm rounded-lg font-medium hover:from-blue-600 hover:to-purple-600 transition-all"
                  >
                    Connect
                  </motion.button>
                ) : provider.status === 'connected' ? (
                  <>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => handleRefresh(provider.name)}
                      className="flex-1 px-3 py-2 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white text-sm rounded-lg font-medium hover:bg-gray-300 dark:hover:bg-gray-600 transition-all"
                    >
                      Refresh
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => console.log(`Managing ${provider.name}...`)}
                      className="px-3 py-2 bg-blue-500 text-white text-sm rounded-lg font-medium hover:bg-blue-600 transition-all"
                    >
                      Manage
                    </motion.button>
                  </>
                ) : (
                  <div className="flex-1 px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 text-sm rounded-lg text-center">
                    Connecting...
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Integration Details */}
      <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg">
        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
          Integration Capabilities
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-blue-500" />
            <span className="text-gray-700 dark:text-gray-300">Compute</span>
          </div>
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-green-500" />
            <span className="text-gray-700 dark:text-gray-300">Storage</span>
          </div>
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-purple-500" />
            <span className="text-gray-700 dark:text-gray-300">Security</span>
          </div>
          <div className="flex items-center gap-2">
            <Cloud className="w-4 h-4 text-orange-500" />
            <span className="text-gray-700 dark:text-gray-300">Networking</span>
          </div>
        </div>
      </div>
    </div>
  )
}