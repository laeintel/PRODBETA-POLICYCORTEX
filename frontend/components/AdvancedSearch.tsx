'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { Search, X, Plus, Filter, Save, History } from 'lucide-react'

interface SearchCondition {
  id: string
  field: string
  operator: string
  value: string
  logicalOperator: 'AND' | 'OR'
}

interface SearchField {
  key: string
  label: string
  type: 'text' | 'number' | 'date' | 'select' | 'boolean'
  operators: string[]
  options?: { label: string; value: any }[]
}

interface SavedSearch {
  id: string
  name: string
  conditions: SearchCondition[]
  createdAt: string
}

interface AdvancedSearchProps {
  fields: SearchField[]
  onSearch: (conditions: SearchCondition[]) => void
  onClose?: () => void
  savedSearches?: SavedSearch[]
  onSaveSearch?: (name: string, conditions: SearchCondition[]) => void
  onLoadSearch?: (search: SavedSearch) => void
  className?: string
}

const DEFAULT_OPERATORS = {
  text: ['contains', 'equals', 'starts with', 'ends with', 'not contains', 'not equals'],
  number: ['equals', 'not equals', 'greater than', 'less than', 'between'],
  date: ['equals', 'before', 'after', 'between'],
  select: ['equals', 'not equals', 'in', 'not in'],
  boolean: ['is', 'is not']
}

export default function AdvancedSearch({
  fields,
  onSearch,
  onClose,
  savedSearches = [],
  onSaveSearch,
  onLoadSearch,
  className = ''
}: AdvancedSearchProps) {
  const [conditions, setConditions] = useState<SearchCondition[]>([
    {
      id: Date.now().toString(),
      field: fields[0]?.key || '',
      operator: fields[0]?.operators[0] || 'contains',
      value: '',
      logicalOperator: 'AND'
    }
  ])
  
  const [searchName, setSearchName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [showSavedSearches, setShowSavedSearches] = useState(false)

  // Add new condition
  const addCondition = useCallback(() => {
    const newCondition: SearchCondition = {
      id: Date.now().toString(),
      field: fields[0]?.key || '',
      operator: fields[0]?.operators[0] || 'contains',
      value: '',
      logicalOperator: 'AND'
    }
    setConditions([...conditions, newCondition])
  }, [conditions, fields])

  // Remove condition
  const removeCondition = useCallback((id: string) => {
    setConditions(conditions.filter(c => c.id !== id))
  }, [conditions])

  // Update condition
  const updateCondition = useCallback((id: string, updates: Partial<SearchCondition>) => {
    setConditions(conditions.map(c => 
      c.id === id ? { ...c, ...updates } : c
    ))
  }, [conditions])

  // Execute search
  const handleSearch = useCallback(() => {
    const validConditions = conditions.filter(c => c.field && c.operator && c.value)
    onSearch(validConditions)
  }, [conditions, onSearch])

  // Save search
  const handleSaveSearch = useCallback(() => {
    if (searchName && onSaveSearch) {
      onSaveSearch(searchName, conditions)
      setSearchName('')
      setShowSaveDialog(false)
    }
  }, [searchName, conditions, onSaveSearch])

  // Load saved search
  const handleLoadSearch = useCallback((search: SavedSearch) => {
    setConditions(search.conditions)
    if (onLoadSearch) {
      onLoadSearch(search)
    }
    setShowSavedSearches(false)
  }, [onLoadSearch])

  // Clear all conditions
  const clearConditions = useCallback(() => {
    setConditions([
      {
        id: Date.now().toString(),
        field: fields[0]?.key || '',
        operator: fields[0]?.operators[0] || 'contains',
        value: '',
        logicalOperator: 'AND'
      }
    ])
  }, [fields])

  // Get field configuration
  const getField = useCallback((key: string) => {
    return fields.find(f => f.key === key)
  }, [fields])

  // Get operators for field
  const getOperators = useCallback((fieldKey: string) => {
    const field = getField(fieldKey)
    if (field?.operators) return field.operators
    return DEFAULT_OPERATORS[field?.type || 'text'] || DEFAULT_OPERATORS.text
  }, [getField])

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Advanced Search</h3>
        <div className="flex items-center space-x-2">
          {savedSearches.length > 0 && (
            <button
              onClick={() => setShowSavedSearches(!showSavedSearches)}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
              title="Saved searches"
            >
              <History className="w-5 h-5" />
            </button>
          )}
          {onSaveSearch && (
            <button
              onClick={() => setShowSaveDialog(!showSaveDialog)}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
              title="Save search"
            >
              <Save className="w-5 h-5" />
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Save Search Dialog */}
      {showSaveDialog && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={searchName}
              onChange={(e) => setSearchName(e.target.value)}
              placeholder="Enter search name..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSaveSearch}
              disabled={!searchName}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              Save
            </button>
            <button
              onClick={() => setShowSaveDialog(false)}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Saved Searches */}
      {showSavedSearches && savedSearches.length > 0 && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Saved Searches</h4>
          <div className="space-y-2">
            {savedSearches.map(search => (
              <button
                key={search.id}
                onClick={() => handleLoadSearch(search)}
                className="w-full text-left px-3 py-2 bg-white border border-gray-200 rounded-md hover:bg-gray-50"
              >
                <div className="font-medium text-gray-900">{search.name}</div>
                <div className="text-sm text-gray-500">
                  {search.conditions.length} conditions • {new Date(search.createdAt).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search Conditions */}
      <div className="space-y-4 mb-6">
        {conditions.map((condition, index) => (
          <div key={condition.id} className="flex items-start space-x-2">
            {/* Logical Operator */}
            {index > 0 && (
              <select
                value={condition.logicalOperator}
                onChange={(e) => updateCondition(condition.id, { 
                  logicalOperator: e.target.value as 'AND' | 'OR' 
                })}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="AND">AND</option>
                <option value="OR">OR</option>
              </select>
            )}
            
            {/* Field */}
            <select
              value={condition.field}
              onChange={(e) => {
                const field = e.target.value
                const operators = getOperators(field)
                updateCondition(condition.id, { 
                  field,
                  operator: operators[0] || 'contains'
                })
              }}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {fields.map(field => (
                <option key={field.key} value={field.key}>
                  {field.label}
                </option>
              ))}
            </select>

            {/* Operator */}
            <select
              value={condition.operator}
              onChange={(e) => updateCondition(condition.id, { operator: e.target.value })}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {getOperators(condition.field).map(op => (
                <option key={op} value={op}>
                  {op}
                </option>
              ))}
            </select>

            {/* Value */}
            {(() => {
              const field = getField(condition.field)
              
              if (field?.type === 'select' && field.options) {
                return (
                  <select
                    value={condition.value}
                    onChange={(e) => updateCondition(condition.id, { value: e.target.value })}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select...</option>
                    {field.options.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                )
              }
              
              if (field?.type === 'boolean') {
                return (
                  <select
                    value={condition.value}
                    onChange={(e) => updateCondition(condition.id, { value: e.target.value })}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select...</option>
                    <option value="true">True</option>
                    <option value="false">False</option>
                  </select>
                )
              }
              
              if (field?.type === 'date') {
                return (
                  <input
                    type="date"
                    value={condition.value}
                    onChange={(e) => updateCondition(condition.id, { value: e.target.value })}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                )
              }
              
              if (field?.type === 'number') {
                return (
                  <input
                    type="number"
                    value={condition.value}
                    onChange={(e) => updateCondition(condition.id, { value: e.target.value })}
                    placeholder="Enter value..."
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                )
              }
              
              return (
                <input
                  type="text"
                  value={condition.value}
                  onChange={(e) => updateCondition(condition.id, { value: e.target.value })}
                  placeholder="Enter value..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )
            })()}

            {/* Remove button */}
            {conditions.length > 1 && (
              <button
                onClick={() => removeCondition(condition.id)}
                className="p-2 text-red-500 hover:text-red-700 hover:bg-red-50 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <button
          onClick={addCondition}
          className="flex items-center space-x-2 px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg"
        >
          <Plus className="w-5 h-5" />
          <span>Add Condition</span>
        </button>

        <div className="flex items-center space-x-2">
          <button
            onClick={clearConditions}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Clear
          </button>
          <button
            onClick={handleSearch}
            className="flex items-center space-x-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Search className="w-5 h-5" />
            <span>Search</span>
          </button>
        </div>
      </div>
    </div>
  )
}