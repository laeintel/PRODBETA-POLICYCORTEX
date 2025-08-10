'use client'

import { useEffect, useState } from 'react'
import { createAction, getAction, streamActionEvents, type CreateActionRequest, type ActionRecord } from '../lib/actions-api'

interface ActionDrawerProps {
  open: boolean
  onClose: () => void
  request: CreateActionRequest | null
}

export default function ActionDrawer({ open, onClose, request }: ActionDrawerProps) {
  const [actionId, setActionId] = useState<string | null>(null)
  const [record, setRecord] = useState<ActionRecord | null>(null)
  const [events, setEvents] = useState<string[]>([])
  const [lastFocused, setLastFocused] = useState<HTMLElement | null>(null)

  useEffect(() => {
    if (!open || !request) return
    let es: { close: () => void } | null = null
    ;(async () => {
      try {
        const { action_id } = await createAction(request)
        setActionId(action_id)
        // initial fetch
        const rec = await getAction(action_id)
        setRecord(rec)
        // stream events
        es = streamActionEvents(action_id, (msg) => setEvents((prev) => [...prev, msg]))
      } catch (e) {
        console.error(e)
      }
    })()
    return () => {
      es?.close()
      setActionId(null)
      setRecord(null)
      setEvents([])
    }
  }, [open, request])

  // Accessibility: trap focus and restore
  useEffect(() => {
    if (open) {
      setLastFocused(document.activeElement as HTMLElement)
      document.body.style.overflow = 'hidden'
      setTimeout(() => {
        const el = document.getElementById('action-drawer-title')
        el?.focus()
      }, 0)
    } else {
      document.body.style.overflow = ''
      lastFocused?.focus()
    }
    return () => { document.body.style.overflow = '' }
  }, [open])

  if (!open || !request) return null

  return (
    <div className="fixed inset-0 z-[100] flex items-end justify-center" aria-modal="true" role="dialog">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-3xl bg-slate-900 border border-white/10 rounded-t-2xl p-4 shadow-2xl focus:outline-none" tabIndex={-1}>
        <div className="flex items-center justify-between">
          <h3 id="action-drawer-title" className="text-white font-semibold" tabIndex={0}>Action</h3>
          <button className="text-gray-300 hover:text-white" onClick={onClose} aria-label="Close action drawer">Close</button>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
          <div className="bg-white/5 rounded p-3">
            <div className="text-gray-300">Summary</div>
            <pre className="text-gray-200 text-xs overflow-auto">{JSON.stringify(request, null, 2)}</pre>
          </div>
          <div className="bg-white/5 rounded p-3">
            <div className="text-gray-300">Status</div>
            <div className="text-white mt-1">{record?.status || 'queued'}</div>
            {actionId && <div className="text-gray-400 text-xs">Action ID: {actionId}</div>}
          </div>
        </div>
        <div className="mt-3 bg-white/5 rounded p-3">
          <div className="text-gray-300 mb-1">Live Progress</div>
          <div className="h-32 overflow-auto text-xs text-gray-200 space-y-1">
            {events.map((e, i) => (
              <div key={i}>• {e}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}


