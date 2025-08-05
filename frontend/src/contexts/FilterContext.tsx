import React, { createContext, useContext, useState, ReactNode } from 'react'

export interface FilterCriteria {
  subscriptions: string[]
  resourceGroups: string[]
  resourceTypes: string[]
  resourceName: string
  managementGroups: string[]
  locations: string[]
  tags: Record<string, string>
}

export interface FilterContextType {
  filters: FilterCriteria
  setFilters: (filters: Partial<FilterCriteria>) => void
  resetFilters: () => void
  isFilterActive: boolean
  applyFilters: <T extends { id: string; name: string; type?: string; resourceGroup?: string; subscription?: string; location?: string; tags?: Record<string, string> }>(items: T[]) => T[]
}

const defaultFilters: FilterCriteria = {
  subscriptions: [],
  resourceGroups: [],
  resourceTypes: [],
  resourceName: '',
  managementGroups: [],
  locations: [],
  tags: {}
}

const FilterContext = createContext<FilterContextType | undefined>(undefined)

export const useFilter = () => {
  const context = useContext(FilterContext)
  if (!context) {
    throw new Error('useFilter must be used within a FilterProvider')
  }
  return context
}

interface FilterProviderProps {
  children: ReactNode
}

export const FilterProvider: React.FC<FilterProviderProps> = ({ children }) => {
  const [filters, setFiltersState] = useState<FilterCriteria>(defaultFilters)

  const setFilters = (newFilters: Partial<FilterCriteria>) => {
    setFiltersState(prev => ({ ...prev, ...newFilters }))
  }

  const resetFilters = () => {
    setFiltersState(defaultFilters)
  }

  const isFilterActive = Object.values(filters).some(value => {
    if (Array.isArray(value)) {
      return value.length > 0
    }
    if (typeof value === 'object' && value !== null) {
      return Object.keys(value).length > 0
    }
    return Boolean(value)
  })

  const applyFilters = <T extends { 
    id: string; 
    name: string; 
    type?: string; 
    resourceGroup?: string; 
    subscription?: string; 
    location?: string; 
    tags?: Record<string, string> 
  }>(items: T[]): T[] => {
    return items.filter(item => {
      // Filter by subscription
      if (filters.subscriptions.length > 0) {
        const itemSubscription = item.subscription || extractSubscriptionFromId(item.id)
        if (!filters.subscriptions.some(sub => itemSubscription?.includes(sub))) {
          return false
        }
      }

      // Filter by resource group
      if (filters.resourceGroups.length > 0) {
        const itemResourceGroup = item.resourceGroup || extractResourceGroupFromId(item.id)
        if (!filters.resourceGroups.some(rg => itemResourceGroup?.toLowerCase().includes(rg.toLowerCase()))) {
          return false
        }
      }

      // Filter by resource type
      if (filters.resourceTypes.length > 0) {
        const itemType = item.type || 'Unknown'
        if (!filters.resourceTypes.some(type => itemType.toLowerCase().includes(type.toLowerCase()))) {
          return false
        }
      }

      // Filter by resource name (partial match)
      if (filters.resourceName.trim()) {
        const searchTerm = filters.resourceName.toLowerCase().trim()
        if (!item.name.toLowerCase().includes(searchTerm)) {
          return false
        }
      }

      // Filter by location
      if (filters.locations.length > 0 && item.location) {
        if (!filters.locations.some(loc => item.location?.toLowerCase().includes(loc.toLowerCase()))) {
          return false
        }
      }

      // Filter by tags
      if (Object.keys(filters.tags).length > 0 && item.tags) {
        const hasMatchingTag = Object.entries(filters.tags).some(([key, value]) => {
          const itemTagValue = item.tags?.[key]
          return itemTagValue && itemTagValue.toLowerCase().includes(value.toLowerCase())
        })
        if (!hasMatchingTag) {
          return false
        }
      }

      return true
    })
  }

  // Helper functions to extract information from Azure resource IDs
  const extractSubscriptionFromId = (id: string): string | undefined => {
    const match = id.match(/\/subscriptions\/([^\/]+)/)
    return match ? match[1] : undefined
  }

  const extractResourceGroupFromId = (id: string): string | undefined => {
    const match = id.match(/\/resourceGroups\/([^\/]+)/)
    return match ? match[1] : undefined
  }

  const contextValue: FilterContextType = {
    filters,
    setFilters,
    resetFilters,
    isFilterActive,
    applyFilters
  }

  return (
    <FilterContext.Provider value={contextValue}>
      {children}
    </FilterContext.Provider>
  )
}