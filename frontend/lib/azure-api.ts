// Azure API integration for real data fetching
import { useState, useEffect } from 'react'

// Use relative URLs to leverage Next.js proxy configuration
const API_BASE = ''

export interface AzureResource {
  id: string
  name: string
  type: string
  resourceGroup: string
  location: string
  tags: Record<string, string>
  status: string
  compliance: string
  cost?: number
  monthlyCost?: number
  savings?: number
  cpu?: number
  memory?: number
  storage?: number
  createdDate?: string
  lastModified?: string
  recommendations?: string[]
}

export interface AzurePolicy {
  id: string
  name: string
  description: string
  category: string
  type: 'BuiltIn' | 'Custom'
  effect: string
  scope: string
  assignments: number
  compliance: {
    compliant: number
    nonCompliant: number
    exempt: number
    percentage: number
  }
  lastModified: string
  createdBy: string
  parameters: Record<string, any>
  resourceTypes: string[]
  status: string
}

export interface CostBreakdown {
  resourceId: string
  resourceName: string
  resourceType: string
  dailyCost: number
  monthlyCost: number
  trend: number
  tags: Record<string, string>
}

export interface RbacAssignment {
  id: string
  principalId: string
  principalName: string
  principalType: string
  roleDefinitionId: string
  roleName: string
  scope: string
  createdDate: string
  lastUsed?: string
}

// Fetch Azure resources (deep insights preferred)
export async function fetchAzureResources(): Promise<AzureResource[]> {
  // Prefer deep endpoint for richer data; fall back to basic
  try {
    const resp = await fetch(`${API_BASE}/api/v1/resources/deep`)
    if (resp.ok) {
      const data = await resp.json()
      const raw = (data && (data.resources ?? data)) as any
      const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
      return items.map((r) => ({
        id: r.id,
        name: r.name,
        type: r.type,
        resourceGroup: r.resourceGroup || (r.id?.split('/')?.[4] ?? ''),
        location: r.location,
        tags: r.tags || {},
        status: r.healthStatus || 'Unknown',
        compliance: r.complianceStatus || 'Unknown',
        monthlyCost: typeof r.costEstimate === 'number' ? Number(r.costEstimate) : 0,
        cost: typeof r.costEstimate === 'number' ? Number(r.costEstimate) / 720 : 0, // approx hourly
        savings: 0,
        cpu: r.utilization?.cpu ?? 0,
        memory: r.utilization?.memory ?? 0,
        storage: r.utilization?.disk ?? 0,
        createdDate: r.createdDate || undefined,
        lastModified: r.lastModified || undefined,
        recommendations: r.recommendations || [],
      })) as AzureResource[]
    }
  } catch (error) {
    console.debug('Deep resource insights unavailable, falling back:', error)
  }

  try {
    const resp = await fetch(`${API_BASE}/api/v1/resources`)
    const data = await resp.json()
    const raw = (data && (data.resources ?? data)) as any
    const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
    return items as AzureResource[]
  } catch (error) {
    console.error('Error fetching Azure resources:', error)
    return []
  }
}

// Fetch Azure policies (deep compliance preferred)
export async function fetchAzurePolicies(): Promise<AzurePolicy[]> {
  // Try deep policy compliance for richer data
  try {
    const resp = await fetch(`${API_BASE}/api/v1/policies/deep`)
    if (resp.ok) {
      const data = await resp.json()
      const raw = (data && (data.complianceResults ?? data)) as any
      const results: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
      return results.map((res) => {
        const a = res.assignment || {}
        const s = res.summary || {}
        return {
          id: a.id || a.name || '',
          name: a.name || '',
          description: a.description || '',
          category: 'Governance',
          type: a.policyDefinitionId?.includes('/providers/Microsoft.Authorization/policyDefinitions/') ? 'BuiltIn' : 'Custom',
          effect: 'Audit',
          scope: a.scope || '',
          assignments: 1,
          compliance: {
            compliant: Number(s.compliantResources || 0),
            nonCompliant: Number(s.nonCompliantResources || 0),
            exempt: 0,
            percentage: Number(s.compliancePercentage || 0),
          },
          lastModified: new Date().toISOString(),
          createdBy: 'Azure',
          parameters: a.parameters || {},
          resourceTypes: [],
          status: 'Active',
        } as AzurePolicy
      })
    }
  } catch (error) {
    console.debug('Deep policy endpoint unavailable, falling back:', error)
  }

  try {
    const resp = await fetch(`${API_BASE}/api/v1/policies`)
    const data = await resp.json()
    const raw = (data && (data.policies ?? data)) as any
    const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
    return items as AzurePolicy[]
  } catch (error) {
    console.error('Error fetching Azure policies:', error)
    return []
  }
}

// Fetch cost breakdown
export async function fetchCostBreakdown(): Promise<CostBreakdown[]> {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/costs/deep`)
    const data = await resp.json()
    const raw = (data && (data.breakdown ?? data)) as any
    const list: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
    const breakdown: CostBreakdown[] = list.map((b: any) => ({
      resourceId: `${b.resourceGroup}-${b.service}`,
      resourceName: b.service,
      resourceType: 'unknown',
      dailyCost: (b.cost || 0) / 30,
      monthlyCost: b.cost || 0,
      trend: 0,
      tags: { ResourceGroup: b.resourceGroup || 'Unknown' }
    }))
    return breakdown
  } catch (error) {
    console.error('Error fetching cost breakdown:', error)
    return []
  }
}

// Fetch RBAC assignments
export async function fetchRbacAssignments(): Promise<RbacAssignment[]> {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/rbac/deep`)
    const data = await resp.json()
    const raw = (data && (data.roleAssignments ?? data)) as any
    const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
    return items.map((a: any) => ({
      id: a.id || a.principalId,
      principalId: a.principalId,
      principalName: a.principalId,
      principalType: a.principalType || 'User',
      roleDefinitionId: a.roleDefinitionId || '',
      roleName: a.roleName || 'Unknown',
      scope: a.scope || '',
      createdDate: new Date().toISOString(),
      lastUsed: undefined,
    }))
  } catch (error) {
    console.error('Error fetching RBAC assignments:', error)
    return []
  }
}

// Hook to fetch Azure resources
export function useAzureResources() {
  const [resources, setResources] = useState<AzureResource[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    fetchAzureResources()
      .then(setResources)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])
  
  return { resources, loading, error }
}

// Hook to fetch Azure policies
export function useAzurePolicies() {
  const [policies, setPolicies] = useState<AzurePolicy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    fetchAzurePolicies()
      .then(setPolicies)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])
  
  return { policies, loading, error }
}

// Hook to fetch cost breakdown
export function useCostBreakdown() {
  const [breakdown, setBreakdown] = useState<CostBreakdown[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    fetchCostBreakdown()
      .then(setBreakdown)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])
  
  return { breakdown, loading, error }
}

// Hook to fetch RBAC assignments
export function useRbacAssignments() {
  const [assignments, setAssignments] = useState<RbacAssignment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  useEffect(() => {
    fetchRbacAssignments()
      .then(setAssignments)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])
  
  return { assignments, loading, error }
}