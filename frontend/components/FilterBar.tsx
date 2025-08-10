'use client'

import { useEffect, useMemo, useState } from 'react'

export type FacetOption = { label: string; value: string }

export type FacetConfig = {
  key: string
  label: string
  placeholder?: string
  options?: FacetOption[]
  allowSearch?: boolean
}

export interface FilterBarProps {
  facets: FacetConfig[]
  initial?: Record<string, string>
  onChange?: (filters: Record<string, string>) => void
}

export default function FilterBar({ facets, initial = {}, onChange }: FilterBarProps) {
  const [filters, setFilters] = useState<Record<string, string>>(() => {
    const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
    const loaded: Record<string, string> = { ...initial }
    facets.forEach((f) => {
      const v = params?.get(f.key)
      if (v) loaded[f.key] = v
    })
    return loaded
  })

  const setFilter = (key: string, value: string) => {
    const updated = { ...filters }
    if (value && value !== 'all') updated[key] = value
    else delete updated[key]
    setFilters(updated)
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    const url = new URL(window.location.href)
    facets.forEach((f) => {
      const v = filters[f.key]
      if (v) url.searchParams.set(f.key, v)
      else url.searchParams.delete(f.key)
    })
    window.history.replaceState(null, '', url.toString())
    onChange?.(filters)
  }, [filters])

  const controls = useMemo(() => facets.map((facet) => {
    const value = filters[facet.key] || 'all'
    return (
      <div key={facet.key} className="flex items-center gap-2">
        <label className="text-xs text-gray-400 min-w-[100px]">{facet.label}</label>
        {facet.options ? (
          <select
            value={value}
            onChange={(e) => setFilter(facet.key, e.target.value)}
            className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
          >
            <option value="all">All</option>
            {facet.options.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        ) : (
          <input
            value={value === 'all' ? '' : value}
            onChange={(e) => setFilter(facet.key, e.target.value)}
            placeholder={facet.placeholder || 'Filter value'}
            className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
          />
        )}
      </div>
    )
  }), [facets, filters])

  return (
    <div className="w-full bg-white/5 border border-white/10 rounded-lg p-3 sticky top-16 z-10 backdrop-blur supports-[backdrop-filter]:bg-white/5">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {controls}
      </div>
    </div>
  )
}


