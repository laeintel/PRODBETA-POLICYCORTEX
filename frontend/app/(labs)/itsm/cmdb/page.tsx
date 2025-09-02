'use client'

import { useState } from 'react'
import { Database, Network, Link, GitBranch, Server, Cloud, Shield, Activity, Search } from 'lucide-react'

interface ConfigurationItem {
  id: string
  name: string
  type: string
  category: 'hardware' | 'software' | 'service' | 'network' | 'database'
  status: 'active' | 'inactive' | 'maintenance' | 'retired'
  relationships: { type: string; target: string }[]
  attributes: Record<string, any>
  lastModified: string
  owner: string
}

export default function CMDBPage() {
  const [configItems] = useState<ConfigurationItem[]>([
    {
      id: 'CI-001',
      name: 'prod-api-gateway',
      type: 'Application',
      category: 'service',
      status: 'active',
      relationships: [
        { type: 'depends-on', target: 'prod-database-01' },
        { type: 'uses', target: 'prod-cache-01' },
        { type: 'hosted-on', target: 'k8s-cluster-01' }
      ],
      attributes: { version: '3.2.0', port: 443, protocol: 'HTTPS' },
      lastModified: '2 hours ago',
      owner: 'Platform Team'
    },
    {
      id: 'CI-002',
      name: 'prod-database-01',
      type: 'Database',
      category: 'database',
      status: 'active',
      relationships: [
        { type: 'hosted-on', target: 'vm-db-01' },
        { type: 'replicated-to', target: 'prod-database-02' }
      ],
      attributes: { engine: 'PostgreSQL', version: '14.5', size: '500GB' },
      lastModified: '1 day ago',
      owner: 'Database Team'
    },
    {
      id: 'CI-003',
      name: 'k8s-cluster-01',
      type: 'Kubernetes Cluster',
      category: 'software',
      status: 'active',
      relationships: [
        { type: 'contains', target: 'namespace-production' },
        { type: 'managed-by', target: 'azure-aks' }
      ],
      attributes: { version: '1.27', nodes: 12, region: 'East US' },
      lastModified: '3 days ago',
      owner: 'DevOps Team'
    },
    {
      id: 'CI-004',
      name: 'firewall-01',
      type: 'Network Device',
      category: 'network',
      status: 'active',
      relationships: [
        { type: 'protects', target: 'prod-subnet-01' },
        { type: 'managed-by', target: 'network-team' }
      ],
      attributes: { vendor: 'Palo Alto', model: 'PA-5220', throughput: '10Gbps' },
      lastModified: '1 week ago',
      owner: 'Network Team'
    }
  ])

  const [selectedCI, setSelectedCI] = useState<ConfigurationItem | null>(null)
  const [searchTerm, setSearchTerm] = useState('')

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'hardware': return <Server className="w-4 h-4" />
      case 'software': return <Cloud className="w-4 h-4" />
      case 'service': return <Activity className="w-4 h-4" />
      case 'network': return <Network className="w-4 h-4" />
      case 'database': return <Database className="w-4 h-4" />
      default: return <GitBranch className="w-4 h-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
      case 'inactive': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
      case 'maintenance': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'retired': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
      default: return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const filteredItems = configItems.filter(item =>
    item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.owner.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Configuration Management Database</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">
            Track configuration items and their relationships
          </p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-muted text-foreground rounded-lg hover:bg-accent transition-colors">
            Import CIs
          </button>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            Add Configuration Item
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-foreground dark:text-white">1,247</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Total CIs</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">1,185</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Active</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">3,842</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Relationships</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">156</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Changes (7d)</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">98.5%</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Accuracy</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* CI List */}
        <div className="col-span-2 bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700">
          <div className="p-4 border-b border-border dark:border-gray-700">
            <div className="flex items-center gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground dark:text-gray-400" />
                <input
                  type="text"
                  placeholder="Search configuration items..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-background dark:bg-gray-900 border border-border dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-muted dark:bg-gray-900">
                <tr>
                  <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">CI Name</th>
                  <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Type</th>
                  <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                  <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Relationships</th>
                  <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Owner</th>
                  <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Modified</th>
                </tr>
              </thead>
              <tbody>
                {filteredItems.map((item) => (
                  <tr
                    key={item.id}
                    className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900 cursor-pointer"
                    onClick={() => setSelectedCI(item)}
                  >
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-muted dark:bg-gray-800 rounded">
                          {getCategoryIcon(item.category)}
                        </div>
                        <div>
                          <div className="font-medium text-foreground dark:text-white">{item.name}</div>
                          <div className="text-xs text-muted-foreground dark:text-gray-400">{item.id}</div>
                        </div>
                      </div>
                    </td>
                    <td className="p-4 text-foreground dark:text-gray-300">{item.type}</td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(item.status)}`}>
                        {item.status}
                      </span>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-1">
                        <Link className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                        <span className="text-sm text-foreground dark:text-gray-300">
                          {item.relationships.length}
                        </span>
                      </div>
                    </td>
                    <td className="p-4 text-foreground dark:text-gray-300">{item.owner}</td>
                    <td className="p-4 text-sm text-muted-foreground dark:text-gray-400">{item.lastModified}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* CI Details */}
        <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-foreground dark:text-white mb-4">
            {selectedCI ? 'Configuration Item Details' : 'Select a CI'}
          </h3>
          
          {selectedCI ? (
            <div className="space-y-4">
              <div>
                <div className="text-sm text-muted-foreground dark:text-gray-400">Name</div>
                <div className="font-medium text-foreground dark:text-white">{selectedCI.name}</div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground dark:text-gray-400">Type</div>
                <div className="font-medium text-foreground dark:text-white">{selectedCI.type}</div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground dark:text-gray-400">Status</div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(selectedCI.status)}`}>
                  {selectedCI.status}
                </span>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground dark:text-gray-400 mb-2">Relationships</div>
                <div className="space-y-2">
                  {selectedCI.relationships.map((rel, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-muted dark:bg-gray-900 rounded">
                      <GitBranch className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                      <span className="text-sm text-foreground dark:text-gray-300">
                        {rel.type} → {rel.target}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground dark:text-gray-400 mb-2">Attributes</div>
                <div className="space-y-1">
                  {Object.entries(selectedCI.attributes).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="text-muted-foreground dark:text-gray-400">{key}:</span>
                      <span className="font-medium text-foreground dark:text-white">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="pt-4 border-t border-border dark:border-gray-700">
                <button className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
                  View Impact Analysis
                </button>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground dark:text-gray-400">
              <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Select a configuration item to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}