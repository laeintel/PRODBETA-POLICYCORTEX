'use client'

import { useState, useEffect } from 'react'
import { Globe, Wifi, Server, Activity, MapPin, Zap, Shield, TrendingUp, AlertTriangle, CheckCircle, Cpu, Database } from 'lucide-react'
import MetricCard from '@/components/MetricCard'
import ChartContainer from '@/components/ChartContainer'

export default function EdgeGovernancePage() {
  const [activeTab, setActiveTab] = useState('network')
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [networkLatency, setNetworkLatency] = useState(12)
  
  useEffect(() => {
    const timer = setInterval(() => {
      setNetworkLatency(prev => Math.max(5, Math.min(25, prev + (Math.random() - 0.5) * 3)))
    }, 2000)
    return () => clearInterval(timer)
  }, [])

  const edgeNodes = [
    {
      id: 'edge-us-west',
      name: 'US West Edge',
      location: 'San Francisco, CA',
      status: 'online',
      latency: '8ms',
      policies: 234,
      violations: 2,
      load: 67,
      region: 'Americas'
    },
    {
      id: 'edge-eu-central',
      name: 'EU Central Edge',
      location: 'Frankfurt, Germany',
      status: 'online',
      latency: '12ms',
      policies: 189,
      violations: 0,
      load: 45,
      region: 'Europe'
    },
    {
      id: 'edge-apac',
      name: 'APAC Edge',
      location: 'Singapore',
      status: 'online',
      latency: '15ms',
      policies: 156,
      violations: 1,
      load: 78,
      region: 'Asia-Pacific'
    },
    {
      id: 'edge-us-east',
      name: 'US East Edge',
      location: 'Virginia, US',
      status: 'degraded',
      latency: '22ms',
      policies: 201,
      violations: 5,
      load: 92,
      region: 'Americas'
    },
    {
      id: 'edge-india',
      name: 'India Edge',
      location: 'Mumbai',
      status: 'online',
      latency: '18ms',
      policies: 134,
      violations: 3,
      load: 56,
      region: 'Asia-Pacific'
    }
  ]

  const edgePolicies = [
    {
      id: 'data-residency',
      name: 'Data Residency Enforcement',
      type: 'Compliance',
      scope: 'Global',
      status: 'active',
      evaluations: '1.2M/day',
      performance: '0.3ms'
    },
    {
      id: 'latency-routing',
      name: 'Latency-Based Routing',
      type: 'Performance',
      scope: 'Regional',
      status: 'active',
      evaluations: '890K/day',
      performance: '0.1ms'
    },
    {
      id: 'gdpr-filter',
      name: 'GDPR Data Filter',
      type: 'Privacy',
      scope: 'EU',
      status: 'active',
      evaluations: '450K/day',
      performance: '0.2ms'
    },
    {
      id: 'threat-detection',
      name: 'Real-Time Threat Detection',
      type: 'Security',
      scope: 'Global',
      status: 'active',
      evaluations: '2.1M/day',
      performance: '0.5ms'
    }
  ]

  const workloads = [
    {
      app: 'Payment Gateway',
      edge: 'Distributed',
      requests: '45K/min',
      p99Latency: '12ms',
      compliance: '99.8%'
    },
    {
      app: 'User Authentication',
      edge: 'Regional',
      requests: '120K/min',
      p99Latency: '8ms',
      compliance: '99.9%'
    },
    {
      app: 'Content Delivery',
      edge: 'Global',
      requests: '890K/min',
      p99Latency: '5ms',
      compliance: '98.5%'
    },
    {
      app: 'IoT Telemetry',
      edge: 'Local',
      requests: '2.1M/min',
      p99Latency: '3ms',
      compliance: '97.2%'
    }
  ]

  const renderContent = () => {
    switch (activeTab) {
      case 'network':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Edge Nodes"
                value="5"
                icon={<Globe className="w-5 h-5 text-blue-500" />}
              />
              <MetricCard
                title="Avg Latency"
                value={`${networkLatency.toFixed(0)}ms`}
                trend={networkLatency < 15 ? 'up' : 'down'}
                icon={<Activity className="w-5 h-5 text-green-500" />}
              />
              <MetricCard
                title="Policy Evaluations"
                value="4.7M"
                trend="up"
                icon={<Shield className="w-5 h-5 text-purple-500" />}
              />
              <MetricCard
                title="Compliance Rate"
                value="98.9%"
                icon={<CheckCircle className="w-5 h-5 text-green-500" />}
              />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Global Edge Network</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {edgeNodes.map((node) => (
                  <div
                    key={node.id}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedNode === node.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
                    }`}
                    onClick={() => setSelectedNode(node.id)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold">{node.name}</h4>
                        <div className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400">
                          <MapPin className="w-3 h-3" />
                          {node.location}
                        </div>
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        node.status === 'online' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                      }`}>
                        {node.status.toUpperCase()}
                      </div>
                    </div>
                    
                    <div className="space-y-2 mt-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Latency</span>
                        <span className="font-medium">{node.latency}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Policies</span>
                        <span className="font-medium">{node.policies}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Violations</span>
                        <span className={`font-medium ${
                          node.violations === 0 ? 'text-green-600' : 'text-red-600'
                        }`}>{node.violations}</span>
                      </div>
                      
                      <div className="mt-2">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-600 dark:text-gray-400">Load</span>
                          <span>{node.load}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              node.load > 80 ? 'bg-red-500' : node.load > 60 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${node.load}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <ChartContainer title="Edge Network Performance">
              <div className="h-64 flex items-center justify-center text-gray-500">
                Real-time latency and throughput visualization
              </div>
            </ChartContainer>
          </div>
        )

      case 'policies':
        return (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Edge Policy Engine</h3>
                <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  Deploy Policy
                </button>
              </div>
              
              <div className="space-y-3">
                {edgePolicies.map((policy) => (
                  <div key={policy.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Shield className={`w-5 h-5 ${
                        policy.type === 'Security' ? 'text-red-500' :
                        policy.type === 'Compliance' ? 'text-blue-500' :
                        policy.type === 'Privacy' ? 'text-purple-500' : 'text-green-500'
                      }`} />
                      <div>
                        <div className="font-medium">{policy.name}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {policy.type} • {policy.scope}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-sm font-medium">{policy.evaluations}</div>
                        <div className="text-xs text-gray-500">Avg: {policy.performance}</div>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                        policy.status === 'active' 
                          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                      }`}>
                        {policy.status.toUpperCase()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg p-6 text-white">
                <h3 className="text-lg font-semibold mb-2">WebAssembly Functions</h3>
                <div className="text-3xl font-bold mb-2">47</div>
                <p className="text-sm opacity-90">
                  Sub-millisecond policy evaluation at the edge
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-green-600 to-teal-600 rounded-lg p-6 text-white">
                <h3 className="text-lg font-semibold mb-2">AI Inference Models</h3>
                <div className="text-3xl font-bold mb-2">12</div>
                <p className="text-sm opacity-90">
                  Real-time threat detection and anomaly analysis
                </p>
              </div>
            </div>
          </div>
        )

      case 'workloads':
        return (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Edge Workload Distribution</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">Application</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">Edge Strategy</th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">Requests</th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">P99 Latency</th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-700 dark:text-gray-300">Compliance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {workloads.map((workload, idx) => (
                      <tr key={idx} className="border-b border-gray-100 dark:border-gray-800">
                        <td className="py-3 px-4">
                          <div className="font-medium">{workload.app}</div>
                        </td>
                        <td className="py-3 px-4">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            workload.edge === 'Global' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400' :
                            workload.edge === 'Regional' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400' :
                            workload.edge === 'Distributed' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                            'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                          }`}>
                            {workload.edge}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-sm">{workload.requests}</td>
                        <td className="py-3 px-4 text-right text-sm">
                          <span className={`font-medium ${
                            parseInt(workload.p99Latency) <= 10 ? 'text-green-600' : 'text-yellow-600'
                          }`}>
                            {workload.p99Latency}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-sm">
                          <span className={`font-medium ${
                            parseFloat(workload.compliance) >= 99 ? 'text-green-600' : 'text-yellow-600'
                          }`}>
                            {workload.compliance}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Edge Compute Optimization</h3>
              <p className="text-sm opacity-90 mb-4">
                AI-driven workload placement reduces latency by 67% and costs by 45%
              </p>
              <button className="px-4 py-2 bg-white text-indigo-600 rounded-lg hover:bg-gray-100 transition-colors">
                View Optimization Report
              </button>
            </div>
          </div>
        )

      case 'monitoring':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricCard
                title="Edge Availability"
                value="99.99%"
                icon={<CheckCircle className="w-5 h-5 text-green-500" />}
              />
              <MetricCard
                title="Failed Evaluations"
                value="0.02%"
                icon={<AlertTriangle className="w-5 h-5 text-yellow-500" />}
              />
              <MetricCard
                title="Cache Hit Rate"
                value="94.3%"
                trend="up"
                icon={<Database className="w-5 h-5 text-blue-500" />}
              />
            </div>

            <ChartContainer title="Edge Network Health">
              <div className="h-64 flex items-center justify-center text-gray-500">
                Real-time health metrics and anomaly detection
              </div>
            </ChartContainer>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Recent Incidents</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <div className="flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 text-yellow-600" />
                    <div>
                      <div className="font-medium">High latency detected - US East</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        P99 latency exceeded 20ms threshold
                      </div>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500">2 hours ago</div>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <div>
                      <div className="font-medium">Auto-scaling completed - APAC</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Added 3 edge nodes to handle increased load
                      </div>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500">5 hours ago</div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg">
              <Globe className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Edge Governance Network
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Distributed policy enforcement at the edge with sub-millisecond latency
              </p>
            </div>
          </div>
        </div>

        <div className="flex gap-2 mb-6 overflow-x-auto">
          {['network', 'policies', 'workloads', 'monitoring'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                activeTab === tab
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {renderContent()}

        <div className="mt-8 p-4 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold">Edge Intelligence Status</h3>
              <p className="text-sm opacity-90 mt-1">
                5 nodes • 4.7M evaluations/day • 0.3ms avg latency • 99.99% uptime
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">OPTIMAL</div>
              <div className="text-xs opacity-75">All systems operational</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}