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

import React from 'react'
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight
} from 'lucide-react'

interface ServerPaginationProps {
  page: number
  pageSize: number
  total: number
  totalPages: number
  onPageChange: (page: number) => void
  onPageSizeChange: (pageSize: number) => void
  pageSizeOptions?: number[]
  showPageSizeSelector?: boolean
  showItemRange?: boolean
  className?: string
}

export default function ServerPagination({
  page,
  pageSize,
  total,
  totalPages,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 25, 50, 100],
  showPageSizeSelector = true,
  showItemRange = true,
  className = ''
}: ServerPaginationProps) {
  const startItem = total > 0 ? (page - 1) * pageSize + 1 : 0
  const endItem = Math.min(page * pageSize, total)
  
  // Generate page numbers to display
  const getPageNumbers = () => {
    const pages: (number | string)[] = []
    const maxVisible = 7
    
    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      if (page <= 3) {
        for (let i = 1; i <= 5; i++) {
          pages.push(i)
        }
        pages.push('...')
        pages.push(totalPages)
      } else if (page >= totalPages - 2) {
        pages.push(1)
        pages.push('...')
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pages.push(i)
        }
      } else {
        pages.push(1)
        pages.push('...')
        for (let i = page - 1; i <= page + 1; i++) {
          pages.push(i)
        }
        pages.push('...')
        pages.push(totalPages)
      }
    }
    
    return pages
  }

  if (total === 0) {
    return null
  }

  return (
    <div className={`flex items-center justify-between px-4 py-3 bg-white/5 backdrop-blur-md border-t border-white/10 sm:px-6 ${className}`}>
      <div className="flex items-center justify-between flex-1">
        {/* Mobile view */}
        <div className="flex justify-between flex-1 sm:hidden">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page === 1}
            className="relative inline-flex items-center px-4 py-2 text-sm font-medium text-gray-200 bg-white/10 border border-white/20 rounded-md hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="flex items-center text-sm text-gray-700">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page === totalPages}
            className="relative inline-flex items-center px-4 py-2 ml-3 text-sm font-medium text-gray-200 bg-white/10 border border-white/20 rounded-md hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>

        {/* Desktop view */}
        <div className="hidden sm:flex sm:items-center sm:justify-between sm:flex-1">
          <div className="flex items-center space-x-4">
            {showItemRange && (
              <p className="text-sm text-gray-300">
                Showing <span className="font-medium">{startItem}</span> to{' '}
                <span className="font-medium">{endItem}</span> of{' '}
                <span className="font-medium">{total}</span> results
              </p>
            )}
            
            {showPageSizeSelector && (
              <div className="flex items-center space-x-2">
                <label htmlFor="page-size" className="text-sm text-gray-300">
                  Show
                </label>
                <select
                  id="page-size"
                  value={pageSize}
                  onChange={(e) => onPageSizeChange(Number(e.target.value))}
                  className="px-2 py-1 text-sm bg-white/10 border border-white/20 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500/50 text-gray-100"
                >
                  {pageSizeOptions.map((size) => (
                    <option key={size} value={size}>
                      {size}
                    </option>
                  ))}
                </select>
                <span className="text-sm text-gray-300">per page</span>
              </div>
            )}
          </div>

          <div>
            <nav className="relative z-0 inline-flex -space-x-px rounded-md shadow-sm" aria-label="Pagination">
              {/* First page */}
              <button
                onClick={() => onPageChange(1)}
                disabled={page === 1}
                className="relative inline-flex items-center px-2 py-2 text-sm font-medium text-gray-300 bg-white/10 border border-white/20 rounded-l-md hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="First page"
              >
                <ChevronsLeft className="w-5 h-5" />
              </button>

              {/* Previous page */}
              <button
                onClick={() => onPageChange(page - 1)}
                disabled={page === 1}
                className="relative inline-flex items-center px-2 py-2 text-sm font-medium text-gray-300 bg-white/10 border border-white/20 hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Previous page"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>

              {/* Page numbers */}
              {getPageNumbers().map((pageNum, index) => {
                if (pageNum === '...') {
                  return (
                  <span
                      key={`ellipsis-${index}`}
                      className="relative inline-flex items-center px-4 py-2 text-sm font-medium text-gray-300 bg-white/10 border border-white/20"
                    >
                      ...
                    </span>
                  )
                }
                
                return (
                  <button
                    key={pageNum}
                    onClick={() => onPageChange(pageNum as number)}
                    className={`relative inline-flex items-center px-4 py-2 text-sm font-medium border ${
                      page === pageNum
                        ? 'z-10 bg-purple-600/30 border-purple-500/50 text-white'
                        : 'bg-white/10 border-white/20 text-gray-300 hover:bg-white/20'
                    }`}
                    aria-current={page === pageNum ? 'page' : undefined}
                  >
                    {pageNum}
                  </button>
                )
              })}

              {/* Next page */}
              <button
                onClick={() => onPageChange(page + 1)}
                disabled={page === totalPages}
                className="relative inline-flex items-center px-2 py-2 text-sm font-medium text-gray-300 bg-white/10 border border-white/20 hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Next page"
              >
                <ChevronRight className="w-5 h-5" />
              </button>

              {/* Last page */}
              <button
                onClick={() => onPageChange(totalPages)}
                disabled={page === totalPages}
                className="relative inline-flex items-center px-2 py-2 text-sm font-medium text-gray-300 bg-white/10 border border-white/20 rounded-r-md hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Last page"
              >
                <ChevronsRight className="w-5 h-5" />
              </button>
            </nav>
          </div>
        </div>
      </div>
    </div>
  )
}