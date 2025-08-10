'use client'

import React, { useState, useCallback, useMemo, useEffect } from 'react'
import VirtualizedTable from './VirtualizedTable'
import ServerPagination from './ServerPagination'
import { useServerPagination } from '@/hooks/useServerPagination'
import { Search, Filter, Download, RefreshCw } from 'lucide-react'

interface DataGridColumn<T> {
  key: keyof T | string
  label: string
  width?: number | string
  render?: (value: any, item: T) => React.ReactNode
  sortable?: boolean
  filterable?: boolean
  filterType?: 'text' | 'select' | 'date' | 'number'
  filterOptions?: { label: string; value: any }[]
}

interface DataGridProps<T> {
  columns: DataGridColumn<T>[]
  fetchData: (params: {
    page: number
    pageSize: number
    sortColumn?: string
    sortDirection?: 'asc' | 'desc'
    filters?: Record<string, any>
    search?: string
  }) => Promise<{ data: T[]; total: number }>
  onRowClick?: (item: T, index: number) => void
  title?: string
  allowExport?: boolean
  allowRefresh?: boolean
  showSearch?: boolean
  showFilters?: boolean
  initialPageSize?: number
  pageSizeOptions?: number[]
  className?: string
}

export default function DataGrid<T extends Record<string, any>>({
  columns,
  fetchData,
  onRowClick,
  title,
  allowExport = true,
  allowRefresh = true,
  showSearch = true,
  showFilters = true,
  initialPageSize = 50,
  pageSizeOptions = [10, 25, 50, 100, 250],
  className = ''
}: DataGridProps<T>) {
  const [data, setData] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [filters, setFilters] = useState<Record<string, any>>({})
  const [sortColumn, setSortColumn] = useState<string | undefined>()
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
  const [showFilterPanel, setShowFilterPanel] = useState(false)

  const pagination = useServerPagination({
    initialPageSize,
    pageSizeOptions
  })

  // Load data
  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const result = await fetchData({
        page: pagination.page,
        pageSize: pagination.pageSize,
        sortColumn,
        sortDirection,
        filters,
        search
      })
      
      setData(result.data)
      pagination.setTotal(result.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
      setData([])
      pagination.setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [pagination.page, pagination.pageSize, sortColumn, sortDirection, filters, search])

  // Load data on mount and when parameters change
  useEffect(() => {
    loadData()
  }, [loadData])

  // Handle sorting
  const handleSort = useCallback((column: string, direction: 'asc' | 'desc') => {
    setSortColumn(column)
    setSortDirection(direction)
  }, [])

  // Handle search with debounce
  const [searchInput, setSearchInput] = useState('')
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(searchInput)
      pagination.goToPage(1) // Reset to first page on search
    }, 300)
    
    return () => clearTimeout(timer)
  }, [searchInput])

  // Handle filter change
  const handleFilterChange = useCallback((key: string, value: any) => {
    setFilters(prev => {
      const newFilters = { ...prev }
      if (value === null || value === undefined || value === '') {
        delete newFilters[key]
      } else {
        newFilters[key] = value
      }
      return newFilters
    })
    pagination.goToPage(1) // Reset to first page on filter change
  }, [])

  // Clear all filters
  const clearFilters = useCallback(() => {
    setFilters({})
    setSearchInput('')
    setSearch('')
    pagination.goToPage(1)
  }, [])

  // Export data
  const handleExport = useCallback(async () => {
    try {
      // Fetch all data for export
      const allData = await fetchData({
        page: 1,
        pageSize: pagination.total || 10000, // Get all records
        sortColumn,
        sortDirection,
        filters,
        search
      })
      
      // Convert to CSV
      const headers = columns.map(col => col.label).join(',')
      const rows = allData.data.map(item => 
        columns.map(col => {
          const value = col.key.includes('.')
            ? col.key.split('.').reduce((obj, key) => obj?.[key], item)
            : item[col.key]
          return JSON.stringify(value ?? '')
        }).join(',')
      )
      
      const csv = [headers, ...rows].join('\n')
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `export-${Date.now()}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }, [columns, fetchData, sortColumn, sortDirection, filters, search, pagination.total])

  // Active filter count
  const activeFilterCount = useMemo(() => {
    return Object.keys(filters).length + (search ? 1 : 0)
  }, [filters, search])

  return (
    <div className={`flex flex-col h-full bg-white rounded-lg shadow ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          {title && <h2 className="text-xl font-semibold text-gray-900">{title}</h2>}
          
          <div className="flex items-center space-x-2">
            {allowRefresh && (
              <button
                onClick={loadData}
                disabled={loading}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
                title="Refresh"
              >
                <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
              </button>
            )}
            
            {allowExport && (
              <button
                onClick={handleExport}
                disabled={loading || data.length === 0}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
                title="Export to CSV"
              >
                <Download className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Search and Filter Bar */}
        <div className="flex items-center space-x-4">
          {showSearch && (
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Search..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
          
          {showFilters && (
            <button
              onClick={() => setShowFilterPanel(!showFilterPanel)}
              className={`flex items-center space-x-2 px-4 py-2 border rounded-lg transition-colors ${
                activeFilterCount > 0
                  ? 'bg-blue-50 border-blue-300 text-blue-700'
                  : 'border-gray-300 text-gray-700 hover:bg-gray-50'
              }`}
            >
              <Filter className="w-5 h-5" />
              <span>Filters</span>
              {activeFilterCount > 0 && (
                <span className="px-2 py-0.5 text-xs bg-blue-600 text-white rounded-full">
                  {activeFilterCount}
                </span>
              )}
            </button>
          )}
          
          {activeFilterCount > 0 && (
            <button
              onClick={clearFilters}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear all
            </button>
          )}
        </div>

        {/* Filter Panel */}
        {showFilterPanel && showFilters && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {columns
                .filter(col => col.filterable)
                .map(col => (
                  <div key={col.key as string}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {col.label}
                    </label>
                    
                    {col.filterType === 'select' && col.filterOptions ? (
                      <select
                        value={filters[col.key as string] ?? ''}
                        onChange={(e) => handleFilterChange(col.key as string, e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="">All</option>
                        {col.filterOptions.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : col.filterType === 'number' ? (
                      <input
                        type="number"
                        value={filters[col.key as string] ?? ''}
                        onChange={(e) => handleFilterChange(col.key as string, e.target.value)}
                        placeholder={`Filter ${col.label}...`}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    ) : col.filterType === 'date' ? (
                      <input
                        type="date"
                        value={filters[col.key as string] ?? ''}
                        onChange={(e) => handleFilterChange(col.key as string, e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    ) : (
                      <input
                        type="text"
                        value={filters[col.key as string] ?? ''}
                        onChange={(e) => handleFilterChange(col.key as string, e.target.value)}
                        placeholder={`Filter ${col.label}...`}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    )}
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>

      {/* Data Table */}
      <div className="flex-1 min-h-0">
        {error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-red-600 mb-2">{error}</p>
              <button
                onClick={loadData}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Retry
              </button>
            </div>
          </div>
        ) : (
          <VirtualizedTable
            data={data}
            columns={columns}
            onRowClick={onRowClick}
            onSort={handleSort}
            sortColumn={sortColumn}
            sortDirection={sortDirection}
            loading={loading}
            overscan={10}
          />
        )}
      </div>

      {/* Pagination */}
      {!error && pagination.total > 0 && (
        <ServerPagination
          page={pagination.page}
          pageSize={pagination.pageSize}
          total={pagination.total}
          totalPages={pagination.totalPages}
          onPageChange={pagination.goToPage}
          onPageSizeChange={pagination.setPageSize}
          pageSizeOptions={pageSizeOptions}
        />
      )}
    </div>
  )
}