'use client'

import React from 'react'

interface PaginationProps {
  page: number
  pageSize: number
  total: number
  onPageChange: (nextPage: number) => void
  onPageSizeChange?: (nextSize: number) => void
}

export default function Pagination({ page, pageSize, total, onPageChange, onPageSizeChange }: PaginationProps) {
  const totalPages = Math.max(1, Math.ceil(total / Math.max(1, pageSize)))
  const start = total === 0 ? 0 : (page - 1) * pageSize + 1
  const end = Math.min(total, page * pageSize)

  const go = (p: number) => {
    const next = Math.min(totalPages, Math.max(1, p))
    if (next !== page) onPageChange(next)
  }

  return (
    <div className="flex items-center justify-between gap-3 p-3 bg-white/5 border-t border-white/10">
      <div className="text-xs text-gray-400">{start}-{end} of {total}</div>
      <div className="flex items-center gap-2">
        <button
          className="px-2 py-1 text-sm text-white/80 bg-white/10 rounded disabled:opacity-40"
          onClick={() => go(1)}
          disabled={page <= 1}
          aria-label="First page"
        >«</button>
        <button
          className="px-2 py-1 text-sm text-white/80 bg-white/10 rounded disabled:opacity-40"
          onClick={() => go(page - 1)}
          disabled={page <= 1}
          aria-label="Previous page"
        >‹</button>
        <div className="text-xs text-gray-300">Page {page} / {totalPages}</div>
        <button
          className="px-2 py-1 text-sm text-white/80 bg-white/10 rounded disabled:opacity-40"
          onClick={() => go(page + 1)}
          disabled={page >= totalPages}
          aria-label="Next page"
        >›</button>
        <button
          className="px-2 py-1 text-sm text-white/80 bg-white/10 rounded disabled:opacity-40"
          onClick={() => go(totalPages)}
          disabled={page >= totalPages}
          aria-label="Last page"
        >»</button>
        {onPageSizeChange && (
          <select
            value={pageSize}
            onChange={(e) => onPageSizeChange(Number(e.target.value))}
            className="ml-2 px-2 py-1 bg-white/10 border border-white/20 rounded text-xs text-white"
            aria-label="Rows per page"
          >
            {[10, 25, 50, 100].map(s => (
              <option key={s} value={s}>{s}/page</option>
            ))}
          </select>
        )}
      </div>
    </div>
  )
}


