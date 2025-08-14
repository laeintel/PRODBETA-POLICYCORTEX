'use client'

import { useEffect, useState } from 'react'
import { useAuthenticatedFetch } from '@/contexts/AuthContext'

interface ExceptionRecord {
  id: string
  resource_id: string
  policy_id: string
  reason: string
  status: string
  expires_at: string
  recertify_at?: string
  created_at: string
}

import AppLayout from '@/components/AppLayout'

export default function ExceptionsPage() {
  const [items, setItems] = useState<ExceptionRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [resourceId, setResourceId] = useState('')
  const [policyId, setPolicyId] = useState('')
  const [reason, setReason] = useState('')
  const authedFetch = useAuthenticatedFetch()

  useEffect(() => {
    const controller = new AbortController()
    const load = async () => {
      try {
        const res = await fetch('/api/v1/exceptions', { signal: controller.signal, headers: { 'Accept-Encoding': 'gzip, br' }})
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const json = await res.json()
        setItems(json.items || [])
      } catch (e: any) {
        if (e.name !== 'AbortError') setError(e.message || 'Failed to load exceptions')
      } finally {
        setLoading(false)
      }
    }
    load()
    return () => controller.abort()
  }, [])

  return (
    <AppLayout>
    <div className="p-6">
      <h1 className="text-2xl font-semibold text-white mb-4">Exceptions</h1>
      <div className="mb-6 grid gap-3 grid-cols-1 md:grid-cols-4">
        <input value={resourceId} onChange={e=>setResourceId(e.target.value)} placeholder="Resource ID" className="px-3 py-2 rounded bg-white/10 text-white placeholder-gray-400" />
        <input value={policyId} onChange={e=>setPolicyId(e.target.value)} placeholder="Policy ID" className="px-3 py-2 rounded bg-white/10 text-white placeholder-gray-400" />
        <input value={reason} onChange={e=>setReason(e.target.value)} placeholder="Reason" className="px-3 py-2 rounded bg-white/10 text-white placeholder-gray-400" />
        <div className="flex gap-2">
          <button
            onClick={async ()=>{
              try{
                const res = await authedFetch('/api/v1/exception',{method:'POST',body:JSON.stringify({resource_id:resourceId,policy_id:policyId,reason})})
                if(!res.ok) throw new Error(`HTTP ${res.status}`)
                setResourceId(''); setPolicyId(''); setReason('');
                const list = await fetch('/api/v1/exceptions').then(r=>r.json())
                setItems(list.items||[])
              }catch(e:any){ setError(e.message||'Failed to create exception') }
            }}
            className="px-4 py-2 rounded bg-blue-600 text-white"
          >Create Exception</button>
          <button
            onClick={async ()=>{
              try{ await authedFetch('/api/v1/exceptions/expire',{method:'POST'}); const list = await fetch('/api/v1/exceptions').then(r=>r.json()); setItems(list.items||[]) }catch(e:any){ setError(e.message||'Expire failed') }
            }}
            className="px-4 py-2 rounded bg-yellow-600 text-white"
          >Expire Past</button>
        </div>
      </div>
      {loading && <div className="text-gray-300">Loading…</div>}
      {error && <div className="text-red-400">{error}</div>}
      {!loading && !error && (
        <table className="min-w-full text-sm text-left text-gray-300">
          <thead className="text-xs uppercase text-gray-400">
            <tr>
              <th className="px-3 py-2">ID</th>
              <th className="px-3 py-2">Resource</th>
              <th className="px-3 py-2">Policy</th>
              <th className="px-3 py-2">Reason</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Expires</th>
            </tr>
          </thead>
          <tbody>
            {items.map((e) => (
              <tr key={e.id} className="border-t border-white/10">
                <td className="px-3 py-2">{e.id}</td>
                <td className="px-3 py-2">{e.resource_id}</td>
                <td className="px-3 py-2">{e.policy_id}</td>
                <td className="px-3 py-2">{e.reason}</td>
                <td className="px-3 py-2">{e.status}</td>
                <td className="px-3 py-2">{new Date(e.expires_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
    </AppLayout>
  )
}
