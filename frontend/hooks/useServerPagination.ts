import { useState, useCallback, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'

interface PaginationState {
  page: number
  pageSize: number
  total: number
  totalPages: number
}

interface UseServerPaginationOptions {
  initialPage?: number
  initialPageSize?: number
  pageSizeOptions?: number[]
  onPageChange?: (page: number, pageSize: number) => void
}

export function useServerPagination({
  initialPage = 1,
  initialPageSize = 50,
  pageSizeOptions = [10, 25, 50, 100, 250, 500],
  onPageChange
}: UseServerPaginationOptions = {}) {
  const searchParams = useSearchParams()
  
  const [paginationState, setPaginationState] = useState<PaginationState>({
    page: parseInt(searchParams.get('page') || String(initialPage)),
    pageSize: parseInt(searchParams.get('pageSize') || String(initialPageSize)),
    total: 0,
    totalPages: 0
  })

  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<any[]>([])

  // Update pagination state
  const updatePagination = useCallback((updates: Partial<PaginationState>) => {
    setPaginationState(prev => {
      const newState = { ...prev, ...updates }
      
      // Recalculate total pages if total or pageSize changed
      if (updates.total !== undefined || updates.pageSize !== undefined) {
        newState.totalPages = Math.ceil(newState.total / newState.pageSize)
      }
      
      // Ensure page is within bounds
      if (newState.page > newState.totalPages && newState.totalPages > 0) {
        newState.page = newState.totalPages
      }
      if (newState.page < 1) {
        newState.page = 1
      }
      
      return newState
    })
  }, [])

  // Go to specific page
  const goToPage = useCallback((page: number) => {
    updatePagination({ page })
    onPageChange?.(page, paginationState.pageSize)
  }, [paginationState.pageSize, updatePagination, onPageChange])

  // Go to next page
  const nextPage = useCallback(() => {
    if (paginationState.page < paginationState.totalPages) {
      goToPage(paginationState.page + 1)
    }
  }, [paginationState.page, paginationState.totalPages, goToPage])

  // Go to previous page
  const prevPage = useCallback(() => {
    if (paginationState.page > 1) {
      goToPage(paginationState.page - 1)
    }
  }, [paginationState.page, goToPage])

  // Go to first page
  const firstPage = useCallback(() => {
    goToPage(1)
  }, [goToPage])

  // Go to last page
  const lastPage = useCallback(() => {
    goToPage(paginationState.totalPages)
  }, [paginationState.totalPages, goToPage])

  // Change page size
  const setPageSize = useCallback((pageSize: number) => {
    // Calculate new page to maintain approximate position
    const currentFirstItem = (paginationState.page - 1) * paginationState.pageSize
    const newPage = Math.floor(currentFirstItem / pageSize) + 1
    
    updatePagination({ pageSize, page: newPage })
    onPageChange?.(newPage, pageSize)
  }, [paginationState.page, paginationState.pageSize, updatePagination, onPageChange])

  // Set total items (usually called after data fetch)
  const setTotal = useCallback((total: number) => {
    updatePagination({ total })
  }, [updatePagination])

  // Fetch data function
  const fetchData = useCallback(async (
    fetcher: (page: number, pageSize: number) => Promise<{ data: any[], total: number }>
  ) => {
    setLoading(true)
    try {
      const result = await fetcher(paginationState.page, paginationState.pageSize)
      setData(result.data)
      setTotal(result.total)
      return result
    } catch (error) {
      console.error('Error fetching paginated data:', error)
      throw error
    } finally {
      setLoading(false)
    }
  }, [paginationState.page, paginationState.pageSize, setTotal])

  // Calculate range of items being displayed
  const getDisplayRange = useCallback(() => {
    const start = (paginationState.page - 1) * paginationState.pageSize + 1
    const end = Math.min(start + paginationState.pageSize - 1, paginationState.total)
    return { start, end }
  }, [paginationState])

  // Generate page numbers for pagination UI
  const getPageNumbers = useCallback((maxVisible: number = 7) => {
    const { page, totalPages } = paginationState
    const pages: (number | string)[] = []
    
    if (totalPages <= maxVisible) {
      // Show all pages
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      // Show first, last, current, and surrounding pages with ellipsis
      const leftOffset = Math.floor((maxVisible - 3) / 2)
      const rightOffset = Math.ceil((maxVisible - 3) / 2)
      
      if (page <= leftOffset + 2) {
        // Near start
        for (let i = 1; i <= maxVisible - 2; i++) {
          pages.push(i)
        }
        pages.push('...')
        pages.push(totalPages)
      } else if (page >= totalPages - rightOffset - 1) {
        // Near end
        pages.push(1)
        pages.push('...')
        for (let i = totalPages - (maxVisible - 3); i <= totalPages; i++) {
          pages.push(i)
        }
      } else {
        // Middle
        pages.push(1)
        pages.push('...')
        for (let i = page - leftOffset + 1; i <= page + rightOffset - 1; i++) {
          pages.push(i)
        }
        pages.push('...')
        pages.push(totalPages)
      }
    }
    
    return pages
  }, [paginationState])

  return {
    // State
    page: paginationState.page,
    pageSize: paginationState.pageSize,
    total: paginationState.total,
    totalPages: paginationState.totalPages,
    loading,
    data,
    
    // Navigation
    goToPage,
    nextPage,
    prevPage,
    firstPage,
    lastPage,
    
    // Configuration
    setPageSize,
    setTotal,
    pageSizeOptions,
    
    // Utilities
    fetchData,
    getDisplayRange,
    getPageNumbers,
    
    // Computed
    hasNext: paginationState.page < paginationState.totalPages,
    hasPrev: paginationState.page > 1,
    isEmpty: paginationState.total === 0,
    isFirstPage: paginationState.page === 1,
    isLastPage: paginationState.page === paginationState.totalPages
  }
}