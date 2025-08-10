'use client'

import React, { useCallback, useRef, useEffect, useState } from 'react'
import { VariableSizeList as List } from 'react-window'
import AutoSizer from 'react-virtualized-auto-sizer'
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'

interface Column<T> {
  key: keyof T | string
  label: string
  width?: number | string
  render?: (value: any, item: T) => React.ReactNode
  sortable?: boolean
}

interface VirtualizedTableProps<T> {
  data: T[]
  columns: Column<T>[]
  rowHeight?: number
  overscan?: number
  onRowClick?: (item: T, index: number) => void
  onSort?: (column: string, direction: 'asc' | 'desc') => void
  sortColumn?: string
  sortDirection?: 'asc' | 'desc'
  loading?: boolean
  loadMore?: () => void
  hasMore?: boolean
  className?: string
}

export default function VirtualizedTable<T extends Record<string, any>>({
  data,
  columns,
  rowHeight = 48,
  overscan = 5,
  onRowClick,
  onSort,
  sortColumn,
  sortDirection,
  loading = false,
  loadMore,
  hasMore = false,
  className = ''
}: VirtualizedTableProps<T>) {
  const listRef = useRef<List>(null)
  const [columnWidths, setColumnWidths] = useState<number[]>([])
  const headerRef = useRef<HTMLDivElement>(null)

  // Calculate column widths
  useEffect(() => {
    const calculateWidths = () => {
      if (!headerRef.current) return
      
      const containerWidth = headerRef.current.offsetWidth
      const fixedWidths = columns.reduce((acc, col) => {
        if (typeof col.width === 'number') {
          return acc + col.width
        }
        return acc
      }, 0)
      
      const flexColumns = columns.filter(col => !col.width || typeof col.width === 'string')
      const remainingWidth = containerWidth - fixedWidths
      const flexWidth = remainingWidth / Math.max(flexColumns.length, 1)
      
      const widths = columns.map(col => {
        if (typeof col.width === 'number') {
          return col.width
        }
        if (typeof col.width === 'string' && col.width.endsWith('%')) {
          const percent = parseFloat(col.width) / 100
          return containerWidth * percent
        }
        return flexWidth
      })
      
      setColumnWidths(widths)
    }
    
    calculateWidths()
    window.addEventListener('resize', calculateWidths)
    return () => window.removeEventListener('resize', calculateWidths)
  }, [columns])

  // Handle sorting
  const handleSort = useCallback((column: string) => {
    if (!onSort) return
    
    if (sortColumn === column) {
      onSort(column, sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      onSort(column, 'asc')
    }
  }, [sortColumn, sortDirection, onSort])

  // Row renderer
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const item = data[index]
    const isLastItem = index === data.length - 1
    
    // Trigger load more when reaching the last item
    useEffect(() => {
      if (isLastItem && hasMore && !loading && loadMore) {
        loadMore()
      }
    }, [isLastItem])
    
    if (!item) {
      return (
        <div style={style} className="flex items-center justify-center text-gray-500">
          {loading ? 'Loading...' : 'No data'}
        </div>
      )
    }
    
    return (
      <div
        style={style}
        className={`flex items-center border-b border-gray-200 hover:bg-gray-50 transition-colors ${
          onRowClick ? 'cursor-pointer' : ''
        }`}
        onClick={() => onRowClick?.(item, index)}
      >
        {columns.map((column, colIndex) => {
          const value = column.key.includes('.')
            ? column.key.split('.').reduce((obj, key) => obj?.[key], item)
            : item[column.key]
          
          return (
            <div
              key={`${index}-${colIndex}`}
              className="px-4 py-2 truncate"
              style={{ width: columnWidths[colIndex] || 'auto' }}
              title={String(value)}
            >
              {column.render ? column.render(value, item) : String(value ?? '')}
            </div>
          )
        })}
      </div>
    )
  }

  // Get item size (for variable height rows in the future)
  const getItemSize = useCallback(() => rowHeight, [rowHeight])

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div
        ref={headerRef}
        className="flex items-center bg-gray-100 border-b border-gray-300 font-semibold text-sm sticky top-0 z-10"
        style={{ minHeight: rowHeight }}
      >
        {columns.map((column, index) => (
          <div
            key={column.key as string}
            className={`px-4 py-2 flex items-center justify-between ${
              column.sortable ? 'cursor-pointer hover:bg-gray-200' : ''
            }`}
            style={{ width: columnWidths[index] || 'auto' }}
            onClick={() => column.sortable && handleSort(column.key as string)}
          >
            <span className="truncate">{column.label}</span>
            {column.sortable && (
              <span className="ml-2 flex-shrink-0">
                {sortColumn === column.key ? (
                  sortDirection === 'asc' ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )
                ) : (
                  <ChevronsUpDown className="w-4 h-4 text-gray-400" />
                )}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Table Body with Virtualization */}
      <div className="flex-1">
        <AutoSizer>
          {({ height, width }) => (
            <List
              ref={listRef}
              height={height}
              width={width}
              itemCount={data.length}
              itemSize={getItemSize}
              overscanCount={overscan}
            >
              {Row}
            </List>
          )}
        </AutoSizer>
      </div>

      {/* Loading indicator */}
      {loading && (
        <div className="flex items-center justify-center p-4 border-t border-gray-200">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading more...</span>
        </div>
      )}
    </div>
  )
}