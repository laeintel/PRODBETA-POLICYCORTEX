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

import React, { useEffect, useState } from 'react'

type RoadmapItem = { id: string; name: string; progress: number; description: string }

export default function RoadmapStatusWidget() {
  const [items, setItems] = useState<RoadmapItem[]>([])
  const [updated, setUpdated] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch('/api/v1/roadmap', { cache: 'no-store' })
        if (!res.ok) throw new Error('Failed to load roadmap')
        const data = await res.json()
        setItems(data.items || [])
        setUpdated(data.last_updated)
      } catch (e: any) {
        setError(e?.message || 'Failed to load')
      }
    }
    load()
    const id = setInterval(load, 30000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="rounded-xl border border-white/10 bg-white/10 backdrop-blur-md p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-white">Roadmap Status</h3>
        {updated && <span className="text-xs text-gray-400">{new Date(updated).toLocaleString()}</span>}
      </div>
      {error ? (
        <div className="text-xs text-red-400">{error}</div>
      ) : (
        <ul className="space-y-2">
          {items.map(item => (
            <li key={item.id} className="text-xs">
              <div className="flex items-center justify-between">
                <span className="text-gray-300">{item.name}</span>
                <span className="text-gray-400">{item.progress}%</span>
              </div>
              <div className="w-full h-1 bg-white/10 rounded mt-1">
                <div className="h-1 rounded bg-purple-500" style={{ width: `${item.progress}%` }} />
              </div>
              <div className="text-[10px] text-gray-400 mt-1">{item.description}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
