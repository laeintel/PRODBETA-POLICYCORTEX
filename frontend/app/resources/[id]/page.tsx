'use client'

import { useMemo } from 'react'
import { useParams, useRouter } from 'next/navigation'
import AppLayout from '../../../components/AppLayout'
import Breadcrumbs from '../../../components/Breadcrumbs'
import { useAzureResources } from '../../../lib/azure-api'
import { MapPin, Pause, Play, RefreshCw, Trash2, Zap } from 'lucide-react'
import ActionDrawer from '../../../components/ActionDrawer'
import { useState } from 'react'
import type { CreateActionRequest } from '../../../lib/actions-api'

export default function ResourceDetailPage() {
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const { resources, loading } = useAzureResources()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerRequest, setDrawerRequest] = useState<CreateActionRequest | null>(null)

  const resource = useMemo(() => {
    if (!resources) return null
    try {
      const decoded = decodeURIComponent(params.id)
      return resources.find(r => r.id === decoded) || null
    } catch {
      return null
    }
  }, [resources, params])

  const runAction = (action: string) => {
    if (!resource) return
    setDrawerRequest({ action_type: action, resource_id: resource.id, params: { name: resource.name } })
    setDrawerOpen(true)
  }

  return (
    <AppLayout>
      <div className="p-6 max-w-5xl mx-auto">
        <Breadcrumbs items={[{ href: '/resources', label: 'Resources' }, { href: '#', label: resource?.name || 'Resource' }]} />
        {loading && <div className="text-gray-300">Loading resource…</div>}
        {!loading && !resource && <div className="text-gray-300">Resource not found.</div>}
        {resource && (
          <div className="space-y-6">
            <div className="bg-white/10 border border-white/20 rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl text-white font-bold">{resource.name}</h1>
                  <p className="text-sm text-gray-400">{resource.type}</p>
                </div>
                <div className="flex items-center gap-2">
                  {resource.status === 'Running' ? (
                    <button onClick={() => runAction('stop')} className="px-3 py-1 bg-white/10 rounded text-gray-200 hover:bg-white/20"><Pause className="w-4 h-4" /></button>
                  ) : (
                    <button onClick={() => runAction('start')} className="px-3 py-1 bg-white/10 rounded text-gray-200 hover:bg-white/20"><Play className="w-4 h-4" /></button>
                  )}
                  <button onClick={() => runAction('restart')} className="px-3 py-1 bg-white/10 rounded text-gray-200 hover:bg-white/20"><RefreshCw className="w-4 h-4" /></button>
                  <button onClick={() => runAction('delete')} className="px-3 py-1 bg-red-600 rounded text-white hover:bg-red-700"><Trash2 className="w-4 h-4" /></button>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-gray-400">Resource Group</div>
                  <div className="text-white">{resource.resourceGroup}</div>
                </div>
                <div>
                  <div className="text-gray-400">Location</div>
                  <div className="text-white flex items-center gap-1"><MapPin className="w-3 h-3" /> {resource.location}</div>
                </div>
                <div>
                  <div className="text-gray-400">Status</div>
                  <div className="text-white">{resource.status}</div>
                </div>
                <div>
                  <div className="text-gray-400">Compliance</div>
                  <div className="text-white">{resource.compliance}</div>
                </div>
              </div>
            </div>
            <div className="bg-white/10 border border-white/20 rounded-xl p-6">
              <h2 className="text-white font-semibold mb-2">Tags</h2>
              <div className="flex flex-wrap gap-2">
                {Object.entries(resource.tags || {}).map(([k, v]) => (
                  <span key={k} className="px-2 py-1 text-xs rounded bg-purple-600/20 text-purple-300">{k}: {v as string}</span>
                ))}
                {Object.keys(resource.tags || {}).length === 0 && (
                  <span className="text-gray-400 text-sm">No tags</span>
                )}
              </div>
            </div>
            {resource.recommendations && resource.recommendations.length > 0 && (
              <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-6">
                <h2 className="text-yellow-300 font-semibold mb-2">AI Recommendations</h2>
                <ul className="list-disc pl-5 text-yellow-200 text-sm space-y-1">
                  {resource.recommendations.map((rec, idx) => <li key={idx}>{rec}</li>)}
                </ul>
                <button onClick={() => runAction('apply-recommendations')} className="mt-3 px-3 py-2 bg-yellow-600/30 text-yellow-200 rounded hover:bg-yellow-600/50 text-sm flex items-center gap-2"><Zap className="w-4 h-4" /> Apply Recommendations</button>
              </div>
            )}
          </div>
        )}
      </div>
      <ActionDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} request={drawerRequest} />
    </AppLayout>
  )
}


