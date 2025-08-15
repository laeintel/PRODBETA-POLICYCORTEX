import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import axios from 'axios'

interface Resource {
  id: string
  name: string
  display_name: string
  resource_type: string
  category: 'Policy' | 'CostManagement' | 'SecurityControls' | 'ComputeStorage' | 'NetworksFirewalls'
  location?: string
  tags: Record<string, string>
  status: {
    state: string
    provisioning_state?: string
    availability: number
    performance_score: number
  }
  health: {
    status: 'Healthy' | 'Degraded' | 'Unhealthy' | 'Unknown'
    issues: Array<{
      severity: 'Critical' | 'High' | 'Medium' | 'Low' | 'Info'
      title: string
      description: string
      affected_components?: string[]
      mitigation?: string
    }>
    recommendations: string[]
  }
  cost_data?: {
    daily_cost: number
    monthly_cost: number
    yearly_projection: number
    cost_trend: {
      type: 'Increasing' | 'Decreasing' | 'Stable'
      value?: number
    }
    optimization_potential: number
    currency: string
  }
  compliance_status: {
    is_compliant: boolean
    compliance_score: number
    violations: Array<{
      policy_id: string
      policy_name: string
      severity: 'Critical' | 'High' | 'Medium' | 'Low' | 'Info'
      description: string
      remediation?: string
    }>
    last_assessment: string
  }
  quick_actions: Array<{
    id: string
    label: string
    icon: string
    action_type: string
    confirmation_required: boolean
    estimated_impact?: string
  }>
  insights: Array<{
    insight_type: string
    title: string
    description: string
    impact: string
    recommendation?: string
    confidence: number
  }>
  last_updated: string
}

interface ResourceSummary {
  total_resources: number
  by_category: Record<string, number>
  by_health: Record<string, number>
  total_daily_cost: number
  compliance_score: number
  critical_issues: number
  optimization_opportunities: number
}

interface ResourceCorrelation {
  id: string
  source_resource: string
  target_resource: string
  correlation_type: string
  strength: number
  impact: string
  insights: Array<{
    title: string
    description: string
    evidence: string[]
    confidence: number
  }>
  recommended_actions: Array<{
    action: string
    priority: number
    expected_outcome: string
    effort_level: string
  }>
}

interface ResourceFilter {
  categories?: string[]
  resource_types?: string[]
  locations?: string[]
  health_status?: string[]
  compliance_only_violations?: boolean
  compliance_min_score?: number
  cost_min_daily?: number
  cost_max_daily?: number
}

interface ResourceState {
  // Data
  resources: Resource[]
  summary: ResourceSummary | null
  correlations: ResourceCorrelation[]
  insights: any[]
  
  // UI State
  loading: boolean
  error: string | null
  selectedResource: Resource | null
  filters: ResourceFilter
  searchQuery: string
  viewMode: 'grid' | 'list' | 'insights'
  
  // Real-time updates
  lastRefresh: Date | null
  autoRefreshEnabled: boolean
  refreshInterval: number // milliseconds
  
  // Actions
  fetchResources: (filter?: ResourceFilter) => Promise<void>
  fetchResourceById: (id: string) => Promise<Resource | null>
  fetchResourcesByCategory: (category: string) => Promise<void>
  fetchCorrelations: () => Promise<void>
  fetchInsights: () => Promise<void>
  executeAction: (resourceId: string, actionId: string, confirmation: boolean) => Promise<void>
  
  // UI Actions
  setSelectedResource: (resource: Resource | null) => void
  setFilters: (filters: ResourceFilter) => void
  setSearchQuery: (query: string) => void
  setViewMode: (mode: 'grid' | 'list' | 'insights') => void
  toggleAutoRefresh: () => void
  clearError: () => void
  
  // Optimistic Updates
  optimisticUpdateResource: (resourceId: string, updates: Partial<Resource>) => void
  
  // WebSocket connection for real-time updates
  connectWebSocket: () => void
  disconnectWebSocket: () => void
  websocket: WebSocket | null
}

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
const API_V2_BASE = `${API_BASE_URL}/api/v2`

// Create the store with persistence and devtools
export const useResourceStore = create<ResourceState>()(
  devtools(
    persist(
      immer((set, get) => ({
        // Initial state
        resources: [],
        summary: null,
        correlations: [],
        insights: [],
        loading: false,
        error: null,
        selectedResource: null,
        filters: {},
        searchQuery: '',
        viewMode: 'grid',
        lastRefresh: null,
        autoRefreshEnabled: true,
        refreshInterval: 30000, // 30 seconds
        websocket: null,

        // Fetch all resources with optional filtering
        fetchResources: async (filter?: ResourceFilter) => {
          set((state) => {
            state.loading = true
            state.error = null
          })

          try {
            const params = new URLSearchParams()
            
            if (filter?.categories?.length) {
              params.append('categories', filter.categories.join(','))
            }
            if (filter?.resource_types?.length) {
              params.append('resource_types', filter.resource_types.join(','))
            }
            if (filter?.locations?.length) {
              params.append('locations', filter.locations.join(','))
            }
            if (filter?.health_status?.length) {
              params.append('health_status', filter.health_status.join(','))
            }
            if (filter?.compliance_only_violations !== undefined) {
              params.append('compliance_only_violations', String(filter.compliance_only_violations))
            }
            if (filter?.compliance_min_score !== undefined) {
              params.append('compliance_min_score', String(filter.compliance_min_score))
            }
            if (filter?.cost_min_daily !== undefined) {
              params.append('cost_min_daily', String(filter.cost_min_daily))
            }
            if (filter?.cost_max_daily !== undefined) {
              params.append('cost_max_daily', String(filter.cost_max_daily))
            }

            const response = await axios.get(`${API_V2_BASE}/resources?${params.toString()}`)
            
            set((state) => {
              state.resources = response.data.resources
              state.summary = response.data.summary
              state.lastRefresh = new Date()
              state.loading = false
            })
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to fetch resources'
              state.loading = false
            })
          }
        },

        // Fetch single resource by ID
        fetchResourceById: async (id: string) => {
          try {
            const response = await axios.get(`${API_V2_BASE}/resources/${id}`)
            return response.data
          } catch (error) {
            console.error('Failed to fetch resource:', error)
            return null
          }
        },

        // Fetch resources by category
        fetchResourcesByCategory: async (category: string) => {
          set((state) => {
            state.loading = true
            state.error = null
          })

          try {
            const response = await axios.get(`${API_V2_BASE}/resources/category/${category}`)
            
            set((state) => {
              state.resources = response.data.resources
              state.summary = response.data.summary
              state.lastRefresh = new Date()
              state.loading = false
            })
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to fetch resources'
              state.loading = false
            })
          }
        },

        // Fetch cross-domain correlations
        fetchCorrelations: async () => {
          try {
            const response = await axios.get(`${API_V2_BASE}/resources/correlations`)
            set((state) => {
              state.correlations = response.data
            })
          } catch (error) {
            console.error('Failed to fetch correlations:', error)
          }
        },

        // Fetch AI insights
        fetchInsights: async () => {
          try {
            const response = await axios.get(`${API_V2_BASE}/resources/insights`)
            set((state) => {
              state.insights = response.data
            })
          } catch (error) {
            console.error('Failed to fetch insights:', error)
          }
        },

        // Execute action on a resource
        executeAction: async (resourceId: string, actionId: string, confirmation: boolean) => {
          try {
            // Optimistic update
            set((state) => {
              const resource = state.resources.find(r => r.id === resourceId)
              if (resource) {
                // Update status based on action
                if (actionId === 'stop') {
                  resource.status.state = 'Stopping'
                } else if (actionId === 'start') {
                  resource.status.state = 'Starting'
                } else if (actionId === 'restart') {
                  resource.status.state = 'Restarting'
                }
              }
            })

            const response = await axios.post(
              `${API_V2_BASE}/resources/${resourceId}/actions`,
              { action_id: actionId, confirmation }
            )

            if (response.data.success) {
              // Refresh the specific resource
              const updatedResource = await get().fetchResourceById(resourceId)
              if (updatedResource) {
                set((state) => {
                  const index = state.resources.findIndex(r => r.id === resourceId)
                  if (index !== -1) {
                    state.resources[index] = updatedResource
                  }
                })
              }
            } else {
              // Revert optimistic update
              await get().fetchResources()
            }
          } catch (error) {
            console.error('Failed to execute action:', error)
            // Revert optimistic update
            await get().fetchResources()
          }
        },

        // UI Actions
        setSelectedResource: (resource) => set((state) => {
          state.selectedResource = resource
        }),

        setFilters: (filters) => set((state) => {
          state.filters = filters
        }),

        setSearchQuery: (query) => set((state) => {
          state.searchQuery = query
        }),

        setViewMode: (mode) => set((state) => {
          state.viewMode = mode
        }),

        toggleAutoRefresh: () => set((state) => {
          state.autoRefreshEnabled = !state.autoRefreshEnabled
        }),

        clearError: () => set((state) => {
          state.error = null
        }),

        // Optimistic update for immediate UI feedback
        optimisticUpdateResource: (resourceId, updates) => set((state) => {
          const resource = state.resources.find(r => r.id === resourceId)
          if (resource) {
            Object.assign(resource, updates)
          }
        }),

        // WebSocket for real-time updates
        connectWebSocket: () => {
          const ws = new WebSocket(`ws://localhost:8080/ws/resources`)
          
          ws.onopen = () => {
            console.log('WebSocket connected for real-time resource updates')
          }

          ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            
            if (data.type === 'resource_update') {
              set((state) => {
                const index = state.resources.findIndex(r => r.id === data.resource.id)
                if (index !== -1) {
                  state.resources[index] = data.resource
                } else {
                  state.resources.push(data.resource)
                }
              })
            } else if (data.type === 'resource_deleted') {
              set((state) => {
                state.resources = state.resources.filter(r => r.id !== data.resource_id)
              })
            } else if (data.type === 'insight') {
              set((state) => {
                state.insights.unshift(data.insight)
              })
            }
          }

          ws.onerror = (error) => {
            console.error('WebSocket error:', error)
          }

          ws.onclose = () => {
            console.log('WebSocket disconnected')
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
              if (get().autoRefreshEnabled) {
                get().connectWebSocket()
              }
            }, 5000)
          }

          set((state) => {
            state.websocket = ws
          })
        },

        disconnectWebSocket: () => {
          const ws = get().websocket
          if (ws) {
            ws.close()
            set((state) => {
              state.websocket = null
            })
          }
        },
      })),
      {
        name: 'resource-store',
        partialize: (state) => ({
          filters: state.filters,
          viewMode: state.viewMode,
          autoRefreshEnabled: state.autoRefreshEnabled,
          refreshInterval: state.refreshInterval,
        }),
      }
    )
  )
)

// Auto-refresh hook
export const useAutoRefresh = () => {
  const { fetchResources, autoRefreshEnabled, refreshInterval } = useResourceStore()

  useEffect(() => {
    if (!autoRefreshEnabled) return

    const interval = setInterval(() => {
      fetchResources()
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [autoRefreshEnabled, refreshInterval, fetchResources])
}

// Export for use in components
import { useEffect } from 'react'