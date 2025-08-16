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

import { useMemo } from 'react'
import { useParams, useRouter } from 'next/navigation'
import AppLayout from '../../../components/AppLayout'
import { useCostBreakdown } from '../../../lib/azure-api'

export default function ServiceCostDetail() {
  const params = useParams<{ service: string }>()
  const router = useRouter()
  const { breakdown, loading } = useCostBreakdown()
  const service = decodeURIComponent(params.service)

  const rows = useMemo(() => (breakdown || []).filter(b => b.resourceName === service), [breakdown, service])
  const monthly = rows.reduce((s,r)=> s + (r.monthlyCost||0), 0)

  return (
    <AppLayout>
      <div className="p-6 max-w-5xl mx-auto">
        <button onClick={() => router.back()} className="text-sm text-gray-300 hover:text-white mb-4">← Back</button>
        <h1 className="text-2xl text-white font-bold mb-2">{service}</h1>
        <p className="text-gray-400 mb-4">Monthly: ${monthly.toFixed(2)}</p>
        {loading && <div className="text-gray-300">Loading…</div>}
        {!loading && rows.length === 0 && <div className="text-gray-300">No costs found.</div>}
        {rows.length > 0 && (
          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Resource Group</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Daily</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Monthly</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {rows.map(r => (
                    <tr key={r.resourceId} className="hover:bg-white/5">
                      <td className="px-6 py-3 text-white text-sm">{r.tags?.ResourceGroup || 'Unknown'}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">${r.dailyCost.toFixed(2)}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">${r.monthlyCost.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  )
}


