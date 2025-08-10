'use client'

import AppLayout from '../../../components/AppLayout'
import Breadcrumbs from '../../../components/Breadcrumbs'
import { useParams, useRouter } from 'next/navigation'

export default function NetworkResourceDetail() {
  const params = useParams<{ resourceId: string }>()
  const router = useRouter()
  const id = decodeURIComponent(params.resourceId)

  return (
    <AppLayout>
      <div className="p-6 max-w-5xl mx-auto">
        <Breadcrumbs items={[{ href: '/network', label: 'Network' }, { href: '#', label: id }]} />
        <div className="bg-white/10 border border-white/20 rounded-xl p-6">
          <h1 className="text-2xl text-white font-bold mb-2">{id}</h1>
          <p className="text-gray-400 mb-4">NSG rule analysis (sample)</p>
          <div className="space-y-2 text-sm">
            <div className="text-gray-300">• Rule: Allow * → High risk (0.0.0.0/0)</div>
            <div className="text-gray-300">• Rule: 3389 exposed → High risk</div>
            <div className="text-gray-300">• Recommendation: Restrict source, remove wildcard</div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}


