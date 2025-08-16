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
import { useRbacAssignments } from '../../../lib/azure-api'

export default function PrincipalDetailPage() {
  const params = useParams<{ principalId: string }>()
  const router = useRouter()
  const { assignments, loading } = useRbacAssignments()
  const principalId = decodeURIComponent(params.principalId)

  const rows = useMemo(() => (assignments || []).filter(a => a.principalId === principalId), [assignments, principalId])

  return (
    <AppLayout>
      <div className="p-6 max-w-5xl mx-auto">
        <Breadcrumbs items={[{ href: '/rbac', label: 'RBAC' }, { href: '#', label: principalId }]} />
        <h1 className="text-2xl text-white font-bold mb-2">{principalId}</h1>
        {loading && <div className="text-gray-300">Loading…</div>}
        {!loading && rows.length === 0 && <div className="text-gray-300">No assignments found.</div>}
        {rows.length > 0 && (
          <div className="bg-white/10 backdrop-blur-md rounded-xl border border-white/20 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Role</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Scope</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Created</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {rows.map(a => (
                    <tr key={a.id} className="hover:bg-white/5">
                      <td className="px-6 py-3 text-white text-sm">{a.roleName}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.principalType}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.scope}</td>
                      <td className="px-6 py-3 text-gray-300 text-sm">{a.createdDate}</td>
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


