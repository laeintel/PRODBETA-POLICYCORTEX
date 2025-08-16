"use client"

import { useState, useEffect, useCallback, useRef } from 'react'
import { FixedSizeList as List } from 'react-window'
import InfiniteLoader from 'react-window-infinite-loader'
import { Loader2 } from 'lucide-react'
import { Skeleton } from '@/components/ui/skeleton'

interface VirtualizedInfiniteScrollProps<T> {
  items: T[]
  hasNextPage: boolean
  isLoading: boolean
  loadMore: () => Promise<void>
  renderItem: (item: T, index: number) => React.ReactNode
  itemHeight: number
  height: number
  width?: number | string
  threshold?: number
  className?: string
  emptyMessage?: string
  errorMessage?: string
  onError?: (error: Error) => void
}

export function VirtualizedInfiniteScroll<T>({
  items,
  hasNextPage,
  isLoading,
  loadMore,
  renderItem,
  itemHeight,
  height,
  width = '100%',
  threshold = 5,
  className,
  emptyMessage = "No items found",
  errorMessage = "Failed to load items",
  onError
}: VirtualizedInfiniteScrollProps<T>) {
  const [error, setError] = useState<Error | null>(null)
  const loaderRef = useRef<InfiniteLoader>(null)

  // Calculate total item count including potential loading placeholders
  const itemCount = hasNextPage ? items.length + 1 : items.length

  // Check if item is loaded
  const isItemLoaded = useCallback((index: number) => {
    return !!items[index]
  }, [items])

  // Load more items with error handling
  const handleLoadMore = useCallback(async () => {
    if (isLoading) return

    try {
      setError(null)
      await loadMore()
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to load more items')
      setError(error)
      onError?.(error)
    }
  }, [isLoading, loadMore, onError])

  // Retry loading on error
  const handleRetry = () => {
    setError(null)
    handleLoadMore()
  }

  // Item renderer for react-window
  const Item = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const item = items[index]

    // Show loading skeleton for items that haven't loaded yet
    if (!item) {
      return (
        <div style={style} className="p-4">
          <ItemSkeleton />
        </div>
      )
    }

    return (
      <div style={style}>
        {renderItem(item, index)}
      </div>
    )
  }, [items, renderItem])

  // Reset loader when items change
  useEffect(() => {
    if (loaderRef.current) {
      loaderRef.current.resetloadMoreItemsCache()
    }
  }, [items])

  // Handle empty state
  if (items.length === 0 && !isLoading && !error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-center">
        <p className="text-muted-foreground">{emptyMessage}</p>
      </div>
    )
  }

  // Handle error state
  if (error && items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-center space-y-4">
        <p className="text-destructive">{errorMessage}</p>
        <button
          onClick={handleRetry}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          Try Again
        </button>
      </div>
    )
  }

  return (
    <div className={className}>
      <InfiniteLoader
        ref={loaderRef}
        isItemLoaded={isItemLoaded}
        itemCount={itemCount}
        loadMoreItems={handleLoadMore}
        threshold={threshold}
      >
        {({ onItemsRendered, ref }) => (
          <List
            ref={ref}
            onItemsRendered={onItemsRendered}
            height={height}
            width={width}
            itemCount={itemCount}
            itemSize={itemHeight}
          >
            {Item}
          </List>
        )}
      </InfiniteLoader>

      {/* Loading indicator at bottom */}
      {isLoading && items.length > 0 && (
        <div className="flex items-center justify-center p-4">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span className="ml-2 text-sm text-muted-foreground">Loading more...</span>
        </div>
      )}

      {/* Error indicator at bottom */}
      {error && items.length > 0 && (
        <div className="flex items-center justify-center p-4 text-destructive">
          <span className="text-sm">{errorMessage}</span>
          <button
            onClick={handleRetry}
            className="ml-2 text-sm underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  )
}

// Skeleton component for loading items
function ItemSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-4 w-1/2" />
      <Skeleton className="h-3 w-full" />
    </div>
  )
}

// Hook for infinite scroll with API integration
export function useInfiniteScroll<T>({
  fetchData,
  pageSize = 20,
  enabled = true
}: {
  fetchData: (page: number, size: number) => Promise<{ items: T[]; hasMore: boolean }>
  pageSize?: number
  enabled?: boolean
}) {
  const [items, setItems] = useState<T[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasNextPage, setHasNextPage] = useState(true)
  const [page, setPage] = useState(0)
  const [error, setError] = useState<Error | null>(null)

  // Load initial data
  useEffect(() => {
    if (enabled && items.length === 0) {
      loadMore()
    }
  }, [enabled])

  const loadMore = useCallback(async () => {
    if (isLoading || !hasNextPage) return

    setIsLoading(true)
    setError(null)

    try {
      const result = await fetchData(page, pageSize)
      
      setItems(prev => [...prev, ...result.items])
      setHasNextPage(result.hasMore)
      setPage(prev => prev + 1)
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to fetch data')
      setError(error)
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [fetchData, page, pageSize, isLoading, hasNextPage])

  const reset = useCallback(() => {
    setItems([])
    setPage(0)
    setHasNextPage(true)
    setError(null)
  }, [])

  const retry = useCallback(() => {
    setError(null)
    loadMore()
  }, [loadMore])

  return {
    items,
    isLoading,
    hasNextPage,
    error,
    loadMore,
    reset,
    retry
  }
}

// Simple infinite scroll component without virtualization
interface SimpleInfiniteScrollProps<T> {
  items: T[]
  hasNextPage: boolean
  isLoading: boolean
  loadMore: () => Promise<void>
  renderItem: (item: T, index: number) => React.ReactNode
  className?: string
  loadingComponent?: React.ReactNode
  emptyComponent?: React.ReactNode
  threshold?: number
}

export function SimpleInfiniteScroll<T>({
  items,
  hasNextPage,
  isLoading,
  loadMore,
  renderItem,
  className,
  loadingComponent,
  emptyComponent,
  threshold = 300
}: SimpleInfiniteScrollProps<T>) {
  const sentinelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const sentinel = sentinelRef.current
    if (!sentinel) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isLoading) {
          loadMore()
        }
      },
      { threshold: 0.1, rootMargin: `${threshold}px` }
    )

    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [hasNextPage, isLoading, loadMore, threshold])

  if (items.length === 0 && !isLoading) {
    return emptyComponent || (
      <div className="text-center text-muted-foreground py-8">
        No items found
      </div>
    )
  }

  return (
    <div className={className}>
      {items.map((item, index) => (
        <div key={index}>
          {renderItem(item, index)}
        </div>
      ))}
      
      {/* Intersection observer sentinel */}
      <div ref={sentinelRef} className="h-1" />
      
      {/* Loading indicator */}
      {isLoading && (
        loadingComponent || (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="ml-2 text-sm text-muted-foreground">
              Loading more...
            </span>
          </div>
        )
      )}
    </div>
  )
}