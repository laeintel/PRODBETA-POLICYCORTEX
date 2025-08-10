'use client'

import AppLayout from '../../components/AppLayout'
import FilterBar from '../../components/FilterBar'
import { ChartCard } from '../../components/ChartCards'
import { Shield } from 'lucide-react'

export default function NetworkPage() {
  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
              <Shield className="w-6 h-6 text-red-400" />
              Network Security
            </h1>
            <p className="text-gray-400">Zero-trust governance</p>
          </div>

          <div className="mb-4">
            <FilterBar facets={[
              { key: 'subscription', label: 'Subscription' },
              { key: 'resourceGroup', label: 'Resource Group' },
              { key: 'location', label: 'Location' },
              { key: 'risk', label: 'Risk', options: [
                { label: 'High', value: 'High' },
                { label: 'Medium', value: 'Medium' },
                { label: 'Low', value: 'Low' },
              ]}
            ]} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
            <ChartCard title="Endpoints" subtitle="Public/Private">
              <div className="text-gray-300 text-sm p-4">Coming soon</div>
            </ChartCard>
            <ChartCard title="Active Threats" subtitle="Last 24h">
              <div className="text-gray-300 text-sm p-4">Coming soon</div>
            </ChartCard>
            <ChartCard title="Blocked" subtitle="Intrusions">
              <div className="text-gray-300 text-sm p-4">Coming soon</div>
            </ChartCard>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Resource</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Location</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Risk</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  <tr className="hover:bg-white/5 cursor-pointer">
                    <td className="px-6 py-3 text-white text-sm">Example NSG</td>
                    <td className="px-6 py-3 text-gray-300 text-sm">Microsoft.Network/networkSecurityGroups</td>
                    <td className="px-6 py-3 text-gray-300 text-sm">eastus</td>
                    <td className="px-6 py-3 text-red-400 text-sm">High</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}


