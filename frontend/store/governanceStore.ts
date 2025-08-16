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

export interface GovernanceMetrics {
  policies: {
    total: number
    active: number
    trend?: number
    compliance_rate?: number
  }
  costs: {
    total: number
    optimized: number
    trend?: number
    optimization_rate?: number
  }
  security: {
    high_risk: number
    medium_risk: number
    low_risk: number
    trend?: number
    risk_score?: number
  }
  resources: {
    total: number
    compliant: number
    non_compliant: number
    utilization_rate?: number
  }
  compliance: {
    overall_score?: number
    policies_compliant: number
    policies_non_compliant: number
  }
}

interface GovernanceStore {
  metrics: GovernanceMetrics | null
  loading: boolean
  error: string | null
  setMetrics: (metrics: GovernanceMetrics) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  updateMetric: (category: keyof GovernanceMetrics, updates: any) => void
}

export const useGovernanceStore = create<GovernanceStore>((set) => ({
  metrics: null,
  loading: false,
  error: null,
  
  setMetrics: (metrics) => set({ metrics }),
  
  setLoading: (loading) => set({ loading }),
  
  setError: (error) => set({ error }),
  
  updateMetric: (category, updates) => set((state) => ({
    metrics: state.metrics ? {
      ...state.metrics,
      [category]: {
        ...state.metrics[category],
        ...updates
      }
    } : null
  }))
}))