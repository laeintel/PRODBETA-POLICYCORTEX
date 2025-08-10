'use client'

import { useMemo } from 'react'
import AppLayout from '../../components/AppLayout'
import { useCostBreakdown } from '../../lib/azure-api'
import { DollarSign } from 'lucide-react'
import { ChartCard, CostTrend, ServiceCostBar } from '../../components/ChartCards'
import FilterBar from '../../components/FilterBar'

export default function CostsPage() {
  const { breakdown, loading } = useCostBreakdown()

  const totalMonthly = (breakdown || []).reduce((s,b)=> s + (b.monthlyCost||0), 0)
  const topServices = useMemo(() => (
    Object.values((breakdown || []).reduce((acc:any,b)=>{
      const key = b.resourceName || 'Other'
      acc[key] = acc[key] || { name: key, monthly: 0 }
      acc[key].monthly += b.monthlyCost || 0
      return acc
    }, {})).sort((a:any,b:any)=> b.monthly - a.monthly).slice(0,8)
  ), [breakdown])

  return (
    <AppLayout>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
              <DollarSign className="w-6 h-6 text-green-400" />
              Cost Management
            </h1>
            <p className="text-gray-400">FinOps automation</p>
          </div>

          <div className="mb-4">
            <FilterBar
              facets={[
                { key: 'subscription', label: 'Subscription' },
                { key: 'resourceGroup', label: 'Resource Group' },
                { key: 'service', label: 'Service' },
                { key: 'location', label: 'Location' },
              ]}
              onChange={() => { /* future server-side filtering */ }}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
            <ChartCard title="Spend Trend" subtitle="Month to date">
              <CostTrend data={[{name:'D1',value:820},{name:'D5',value:790},{name:'D10',value:760},{name:'D15',value:740},{name:'D20',value:710},{name:'D25',value:700},{name:'D30',value:690}]} />
            </ChartCard>
            <ChartCard title="Top Services" subtitle="Monthly">
              <ServiceCostBar data={topServices as any} />
            </ChartCard>
            <div className="p-4 bg-white/10 backdrop-blur-md rounded-xl border border-white/20">
              <p className="text-xs text-gray-400">Total Monthly</p>
              <p className="text-3xl font-bold text-white">${totalMonthly.toFixed(2)}</p>
              <p className="text-xs text-gray-400 mt-2">Data sourced from deep cost endpoint</p>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden sticky top-2">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Resource Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Daily</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Monthly</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Trend</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {(loading ? [] : breakdown).map((r) => (
                    <tr key={r.resourceId} className="hover:bg-white/5 cursor-pointer" onClick={() => {
                      if (typeof window !== 'undefined') {
                        window.location.href = `/costs/${encodeURIComponent(r.resourceName)}`
                      }
                    }}>
                      <td className="px-6 py-3 text-white text-sm">{r.resourceName}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">${r.dailyCost.toFixed(2)}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">${r.monthlyCost.toFixed(2)}</td>
                      <td className={`px-6 py-3 text-sm ${r.trend > 0 ? 'text-red-300' : r.trend < 0 ? 'text-green-300' : 'text-gray-300'}`}>{r.trend}%</td>
                    </tr>
                  ))}
                  {(!loading && (breakdown || []).length === 0) && (
                    <tr><td className="px-6 py-6 text-sm text-gray-400" colSpan={4}>No cost data found</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}


