/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import axios from 'axios'
import { api } from '@/lib/api-client'

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
            const payload: any = {}
            if (filter?.categories?.length) payload.categories = filter.categories.join(',')
            if (filter?.resource_types?.length) payload.resource_types = filter.resource_types.join(',')
            if (filter?.locations?.length) payload.locations = filter.locations.join(',')
            if (filter?.health_status?.length) payload.health_status = filter.health_status.join(',')
            if (filter?.compliance_only_violations !== undefined) payload.compliance_only_violations = filter.compliance_only_violations
            if (filter?.compliance_min_score !== undefined) payload.compliance_min_score = filter.compliance_min_score
            if (filter?.cost_min_daily !== undefined) payload.cost_min_daily = filter.cost_min_daily
            if (filter?.cost_max_daily !== undefined) payload.cost_max_daily = filter.cost_max_daily

            const response = await api.getResources(payload)
            const data = response.data as any
            set((state) => {
              state.resources = data?.resources || data || []
              state.summary = data?.summary || null
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
            const response = await api.getResourceDetails(id)
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
            const response = await api.getResources({ categories: category })
            const data = response.data as any
            set((state) => {
              state.resources = data?.resources || data || []
              state.summary = data?.summary || null
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

        // Fetch cross-domain correlations (unified API v1)
        fetchCorrelations: async () => {
          try {
            const resp = await api.getCorrelations()
            if (!resp.error) {
              set((state) => {
                state.correlations = (resp.data as any) || []
              })
            }
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

        // Execute action on a resource (wired to /api/v1/actions with streaming)
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

            const created = await api.createAction(resourceId, actionId, { confirmation })
            if (created.error || created.status >= 400) {
              await get().fetchResources()
              return
            }
            const actionIdCreated = created.data?.action_id || created.data?.id
            if (actionIdCreated) {
              const stop = api.streamActionEvents(String(actionIdCreated), async (msg) => {
                // Optionally parse and react to progress events here
                // console.log('[action-event]', actionIdCreated, msg)
              })
              // Stop the stream after 60s; refresh resource once to reflect final state
              setTimeout(async () => {
                stop()
                const updatedResource = await get().fetchResourceById(resourceId)
                if (updatedResource) {
                  set((state) => {
                    const index = state.resources.findIndex(r => r.id === resourceId)
                    if (index !== -1) {
                      state.resources[index] = updatedResource
                    }
                  })
                }
              }, 60000)
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