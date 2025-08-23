'use client'

import { useState } from 'react'
import { Package, Server, HardDrive, Monitor, Smartphone, Router, DollarSign, Calendar } from 'lucide-react'

export default function AssetsPage() {
  const [assets] = useState([
    { id: 'AST-001', name: 'Dell PowerEdge R740', type: 'Server', status: 'in-use', location: 'DC-East-01', owner: 'Infrastructure', purchaseDate: '2023-01-15', cost: 12500, warranty: 'Active' },
    { id: 'AST-002', name: 'Cisco Catalyst 9300', type: 'Network', status: 'in-use', location: 'DC-East-01', owner: 'Network Team', purchaseDate: '2023-03-20', cost: 8900, warranty: 'Active' },
    { id: 'AST-003', name: 'NetApp FAS8200', type: 'Storage', status: 'in-use', location: 'DC-West-01', owner: 'Storage Team', purchaseDate: '2022-11-10', cost: 45000, warranty: 'Active' },
    { id: 'AST-004', name: 'MacBook Pro M3', type: 'Laptop', status: 'assigned', location: 'Remote', owner: 'John Doe', purchaseDate: '2023-09-05', cost: 3500, warranty: 'Active' },
    { id: 'AST-005', name: 'HP EliteDesk 800', type: 'Desktop', status: 'available', location: 'Office-NYC', owner: 'IT Dept', purchaseDate: '2023-06-12', cost: 1200, warranty: 'Expired' }
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'in-use': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
      case 'assigned': return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
      case 'available': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'maintenance': return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400'
      case 'retired': return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
      default: return 'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'Server': return <Server className="w-4 h-4" />
      case 'Storage': return <HardDrive className="w-4 h-4" />
      case 'Network': return <Router className="w-4 h-4" />
      case 'Desktop': return <Monitor className="w-4 h-4" />
      case 'Laptop': return <Monitor className="w-4 h-4" />
      case 'Mobile': return <Smartphone className="w-4 h-4" />
      default: return <Package className="w-4 h-4" />
    }
  }

  const totalValue = assets.reduce((sum, asset) => sum + asset.cost, 0)

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground dark:text-white">Asset Management</h1>
          <p className="text-muted-foreground dark:text-gray-400 mt-1">Track and manage IT assets lifecycle</p>
        </div>
        <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
          Add Asset
        </button>
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-foreground dark:text-white">247</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Total Assets</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">185</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">In Use</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">32</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Available</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            ${(totalValue / 1000).toFixed(0)}K
          </div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Total Value</div>
        </div>
        <div className="bg-card dark:bg-gray-800 rounded-lg p-4 border border-border dark:border-gray-700">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">18</div>
          <div className="text-sm text-muted-foreground dark:text-gray-400">Warranty Expiring</div>
        </div>
      </div>

      <div className="bg-card dark:bg-gray-800 rounded-lg border border-border dark:border-gray-700">
        <div className="p-4 border-b border-border dark:border-gray-700">
          <h2 className="text-lg font-semibold text-foreground dark:text-white">Asset Inventory</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-muted dark:bg-gray-900">
              <tr>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Asset ID</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Name</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Type</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Status</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Location</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Owner</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Value</th>
                <th className="p-4 text-left text-sm font-medium text-foreground dark:text-white">Warranty</th>
              </tr>
            </thead>
            <tbody>
              {assets.map((asset) => (
                <tr key={asset.id} className="border-b border-border dark:border-gray-700 hover:bg-muted dark:hover:bg-gray-900">
                  <td className="p-4 font-medium text-foreground dark:text-white">{asset.id}</td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 bg-muted dark:bg-gray-800 rounded">
                        {getTypeIcon(asset.type)}
                      </div>
                      <span className="text-foreground dark:text-gray-300">{asset.name}</span>
                    </div>
                  </td>
                  <td className="p-4 text-foreground dark:text-gray-300">{asset.type}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(asset.status)}`}>
                      {asset.status.replace('-', ' ')}
                    </span>
                  </td>
                  <td className="p-4 text-foreground dark:text-gray-300">{asset.location}</td>
                  <td className="p-4 text-foreground dark:text-gray-300">{asset.owner}</td>
                  <td className="p-4 font-medium text-foreground dark:text-white">${asset.cost.toLocaleString()}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      asset.warranty === 'Active' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                      'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {asset.warranty}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}