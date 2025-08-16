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
import Breadcrumbs from '../../../components/Breadcrumbs'
import { useAzurePolicies, type AzurePolicy } from '../../../lib/azure-api'

export default function PolicyDetailPage() {
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const { policies, loading } = useAzurePolicies()

  const policy: AzurePolicy | null = useMemo(() => {
    try {
      const decoded = decodeURIComponent(params.id)
      return (policies || []).find(p => p.id === decoded || p.name === decoded) || null
    } catch {
      return null
    }
  }, [policies, params])

  return (
    <AppLayout>
      <div className="p-6 max-w-5xl mx-auto">
        <Breadcrumbs items={[{ href: '/policies', label: 'Policies' }, { href: '#', label: policy?.name || 'Policy' }]} />
        {loading && <div className="text-gray-300">Loading policy…</div>}
        {!loading && !policy && <div className="text-gray-300">Policy not found.</div>}
        {policy && (
          <div className="space-y-6">
            <div className="bg-white/10 border border-white/20 rounded-xl p-6">
              <h1 className="text-2xl text-white font-bold mb-1">{policy.name}</h1>
              <p className="text-sm text-gray-400 mb-4">{policy.description || 'No description'}</p>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-400">Category</div>
                  <div className="text-white">{policy.category}</div>
                </div>
                <div>
                  <div className="text-gray-400">Effect</div>
                  <div className="text-white">{policy.effect}</div>
                </div>
                <div>
                  <div className="text-gray-400">Type</div>
                  <div className="text-white">{policy.type}</div>
                </div>
                <div>
                  <div className="text-gray-400">Status</div>
                  <div className="text-white">{policy.status}</div>
                </div>
              </div>
            </div>

            <div className="bg-white/10 border border-white/20 rounded-xl p-6">
              <h2 className="text-white font-semibold mb-2">Compliance</h2>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-2xl text-green-400 font-bold">{policy.compliance?.compliant || 0}</p>
                  <p className="text-xs text-gray-400">Compliant</p>
                </div>
                <div>
                  <p className="text-2xl text-red-400 font-bold">{policy.compliance?.nonCompliant || 0}</p>
                  <p className="text-xs text-gray-400">Non-Compliant</p>
                </div>
                <div>
                  <p className="text-2xl text-blue-400 font-bold">{(policy.compliance?.percentage || 0).toFixed(1)}%</p>
                  <p className="text-xs text-gray-400">Overall</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  )
}


