"use client"

import { useState, useEffect, useCallback } from 'react'
import { Search, Filter, X, Save, Clock, Star } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { DatePicker } from '@/components/ui/date-picker'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

interface SearchFilter {
  field: string
  operator: 'equals' | 'contains' | 'starts_with' | 'ends_with' | 'greater_than' | 'less_than' | 'between'
  value: any
  label?: string
}

interface SavedSearch {
  id: string
  name: string
  query: string
  filters: SearchFilter[]
  createdAt: Date
  favorite?: boolean
}

interface AdvancedSearchProps {
  onSearch: (query: string, filters: SearchFilter[]) => void
  placeholder?: string
  fields?: Array<{ key: string; label: string; type: 'text' | 'number' | 'date' | 'select'; options?: string[] }>
  savedSearches?: SavedSearch[]
  onSaveSearch?: (search: Omit<SavedSearch, 'id' | 'createdAt'>) => void
  onLoadSearch?: (search: SavedSearch) => void
  className?: string
}

export function AdvancedSearchV2({
  onSearch,
  placeholder = "Search...",
  fields = [],
  savedSearches = [],
  onSaveSearch,
  onLoadSearch,
  className
}: AdvancedSearchProps) {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<SearchFilter[]>([])
  const [showFilters, setShowFilters] = useState(false)
  const [recentSearches, setRecentSearches] = useState<string[]>([])

  // Debounced search
  const [debouncedQuery, setDebouncedQuery] = useState(query)
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300)
    return () => clearTimeout(timer)
  }, [query])

  useEffect(() => {
    if (debouncedQuery || filters.length > 0) {
      onSearch(debouncedQuery, filters)
    }
  }, [debouncedQuery, filters, onSearch])

  const addFilter = (field: string) => {
    const fieldConfig = fields.find(f => f.key === field)
    if (!fieldConfig) return

    const newFilter: SearchFilter = {
      field,
      operator: fieldConfig.type === 'text' ? 'contains' : 'equals',
      value: '',
      label: fieldConfig.label
    }

    setFilters(prev => [...prev, newFilter])
  }

  const updateFilter = (index: number, updates: Partial<SearchFilter>) => {
    setFilters(prev => 
      prev.map((filter, i) => i === index ? { ...filter, ...updates } : filter)
    )
  }

  const removeFilter = (index: number) => {
    setFilters(prev => prev.filter((_, i) => i !== index))
  }

  const clearAll = () => {
    setQuery('')
    setFilters([])
  }

  const saveCurrentSearch = () => {
    if (!onSaveSearch || (!query && filters.length === 0)) return

    const name = prompt('Enter a name for this search:')
    if (!name) return

    onSaveSearch({
      name,
      query,
      filters,
      favorite: false
    })
  }

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    // Add to recent searches
    if (query && !recentSearches.includes(query)) {
      setRecentSearches(prev => [query, ...prev.slice(0, 4)])
    }
    
    onSearch(query, filters)
  }

  const renderFilterValue = (filter: SearchFilter, index: number) => {
    const fieldConfig = fields.find(f => f.key === filter.field)
    if (!fieldConfig) return null

    switch (fieldConfig.type) {
      case 'select':
        return (
          <Select value={filter.value} onChange={(e) => updateFilter(index, { value: e.target.value })}>
            {fieldConfig.options?.map(option => (
              <SelectItem key={option} value={option}>{option}</SelectItem>
            ))}
          </Select>
        )
      
      case 'date':
        return (
          <DatePicker
            value={filter.value}
            onChange={(date) => updateFilter(index, { value: date })}
          />
        )
      
      case 'number':
        return (
          <Input
            type="number"
            value={filter.value}
            onChange={(e) => updateFilter(index, { value: e.target.value })}
            className="w-32"
          />
        )
      
      default:
        return (
          <Input
            value={filter.value}
            onChange={(e) => updateFilter(index, { value: e.target.value })}
            className="w-32"
            placeholder="Enter value..."
          />
        )
    }
  }

  return (
    <div className={className}>
      <Card>
        <CardContent className="pt-6">
          {/* Main search bar */}
          <form onSubmit={handleSearchSubmit} className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={placeholder}
                className="pl-10 pr-32"
              />
              
              {/* Action buttons */}
              <div className="absolute right-2 top-2 flex gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowFilters(!showFilters)}
                  className={showFilters ? 'bg-muted' : ''}
                >
                  <Filter className="h-4 w-4" />
                </Button>
                
                {savedSearches.length > 0 && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm">
                        <Clock className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-64">
                      <DropdownMenuLabel>Saved Searches</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      {savedSearches.map(search => (
                        <DropdownMenuItem
                          key={search.id}
                          onClick={() => onLoadSearch?.(search)}
                          className="flex items-center justify-between"
                        >
                          <span className="truncate">{search.name}</span>
                          {search.favorite && <Star className="h-3 w-3 fill-current" />}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}

                {onSaveSearch && (query || filters.length > 0) && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={saveCurrentSearch}
                  >
                    <Save className="h-4 w-4" />
                  </Button>
                )}

                {(query || filters.length > 0) && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={clearAll}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            {/* Recent searches */}
            {recentSearches.length > 0 && !query && (
              <div className="flex flex-wrap gap-2">
                <span className="text-sm text-muted-foreground">Recent:</span>
                {recentSearches.map((recent, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => setQuery(recent)}
                    className="h-6 text-xs"
                  >
                    {recent}
                  </Button>
                ))}
              </div>
            )}
          </form>

          {/* Advanced filters */}
          {showFilters && (
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">Filters</h4>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      Add Filter
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    {fields.map(field => (
                      <DropdownMenuItem
                        key={field.key}
                        onClick={() => addFilter(field.key)}
                      >
                        {field.label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {/* Active filters */}
              <div className="space-y-2">
                {filters.map((filter, index) => (
                  <div key={index} className="flex items-center gap-2 p-2 bg-muted/50 rounded-lg">
                    <Badge variant="secondary" className="shrink-0">
                      {filter.label || filter.field}
                    </Badge>
                    
                    <Select
                      value={filter.operator}
                      onChange={(e) => updateFilter(index, { operator: e.target.value as any })}
                    >
                      <SelectItem value="equals">equals</SelectItem>
                      <SelectItem value="contains">contains</SelectItem>
                      <SelectItem value="starts_with">starts with</SelectItem>
                      <SelectItem value="ends_with">ends with</SelectItem>
                      <SelectItem value="greater_than">greater than</SelectItem>
                      <SelectItem value="less_than">less than</SelectItem>
                      <SelectItem value="between">between</SelectItem>
                    </Select>

                    {renderFilterValue(filter, index)}

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFilter(index)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Active filter summary */}
          {filters.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {filters.map((filter, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {filter.label}: {filter.operator} {filter.value}
                  <X 
                    className="ml-1 h-3 w-3 cursor-pointer" 
                    onClick={() => removeFilter(index)}
                  />
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}