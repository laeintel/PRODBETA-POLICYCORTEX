export type AzureOpenAIConfig = {
  endpoint: string;
  apiKey?: string;
  apiVersion?: string;
  deployment: string; // e.g., 'chat-dev'
};

export async function callChat(config: AzureOpenAIConfig, messages: { role: string; content: string }[], options: { temperature?: number } = {}) {
  const url = `${config.endpoint}/openai/deployments/${config.deployment}/chat/completions?api-version=${config.apiVersion || '2024-08-01-preview'}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(config.apiKey ? { 'api-key': config.apiKey } : {}),
    },
    body: JSON.stringify({ messages, temperature: options.temperature ?? 0.2 }),
  });
  if (!res.ok) throw new Error(`Azure OpenAI error: ${res.status}`);
  const json = await res.json();
  return json.choices?.[0]?.message?.content as string;
}
// Azure API integration for real data fetching
import { useState, useEffect } from 'react'
import { useAuthenticatedFetch } from '../contexts/AuthContext'

// Use relative URLs to leverage Next.js proxy configuration
const API_BASE = ''
const DISABLE_DEEP = process.env.NEXT_PUBLIC_DISABLE_DEEP === 'true'

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
export async function fetchAzureResources(fetcher?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>): Promise<AzureResource[]> {
  // Prefer deep endpoint for richer data; fall back to basic
  if (!DISABLE_DEEP) {
    try {
      const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/resources/deep`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/resources/deep`))
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
  }

  try {
    const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/resources`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/resources`))
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
export async function fetchAzurePolicies(fetcher?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>): Promise<AzurePolicy[]> {
  // Try deep policy compliance for richer data
  try {
    const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/policies/deep`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/policies/deep`))
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
    const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/policies`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/policies`))
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    const raw = (data && (data.policies ?? data)) as any
    const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
    return items as AzurePolicy[]
  } catch (error) {
    console.warn('Basic policy endpoint failed, falling back to deep:', error)
    try {
      const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/policies/deep`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/policies/deep`))
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
    } catch (e2) {
      console.error('Error fetching Azure policies:', e2)
      return []
    }
  }
}

// Fetch cost breakdown
export async function fetchCostBreakdown(fetcher?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>): Promise<CostBreakdown[]> {
  try {
    const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/costs/deep`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/costs/deep`))
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
export async function fetchRbacAssignments(fetcher?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>): Promise<RbacAssignment[]> {
  // First try deep endpoint (returns mock structure if Azure not connected)
  try {
    const resp = await (fetcher ? fetcher(`${API_BASE}/api/v1/rbac/deep`, { cache: 'no-store' }) : fetch(`${API_BASE}/api/v1/rbac/deep`))
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    const raw = (data && (data.roleAssignments ?? data)) as any
    const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
    if (items.length > 0) {
      return items.map((a: any) => ({
        id: a.id || a.principalId,
        principalId: a.principalId,
        principalName: a.principalName || a.principalId,
        principalType: a.principalType || 'User',
        roleDefinitionId: a.roleDefinitionId || '',
        roleName: a.roleName || 'Unknown',
        scope: a.scope || '',
        createdDate: new Date().toISOString(),
        lastUsed: undefined,
      }))
    }
  } catch (error) {
    console.warn('Deep RBAC endpoint unavailable, will use fallback:', error)
  }

  // Fallback: minimal mock dataset so UI is populated during demos/offline
  const nowIso = new Date().toISOString()
  const fallback: RbacAssignment[] = [
    {
      id: 'rbac-1',
      principalId: 'user1@contoso.com',
      principalName: 'user1@contoso.com',
      principalType: 'User',
      roleDefinitionId: 'Owner',
      roleName: 'Owner',
      scope: '/subscriptions/demo',
      createdDate: nowIso,
      lastUsed: nowIso,
    },
    {
      id: 'rbac-2',
      principalId: 'spn-ae-devops',
      principalName: 'spn-ae-devops',
      principalType: 'ServicePrincipal',
      roleDefinitionId: 'Contributor',
      roleName: 'Contributor',
      scope: '/subscriptions/demo/resourceGroups/rg-cortex-dev',
      createdDate: nowIso,
      lastUsed: undefined,
    },
    {
      id: 'rbac-3',
      principalId: 'reader-ops',
      principalName: 'reader-ops',
      principalType: 'User',
      roleDefinitionId: 'Reader',
      roleName: 'Reader',
      scope: '/subscriptions/demo',
      createdDate: nowIso,
      lastUsed: undefined,
    },
  ]
  return fallback
}

// Hook to fetch Azure resources
export function useAzureResources() {
  const [resources, setResources] = useState<AzureResource[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isUsingFallback, setIsUsingFallback] = useState(false)
  const authenticatedFetch = useAuthenticatedFetch()
  
  useEffect(() => {
    const load = async () => {
      try {
        // Try deep first for richer data
        try {
          const resp = await (authenticatedFetch as any)(`${API_BASE}/api/v1/resources/deep`, { cache: 'no-store' })
          if (resp && resp.ok) {
            const data = await resp.json()
            const raw = (data && (data.resources ?? data)) as any
            const items: any[] = Array.isArray(raw) ? raw : (raw ? [raw] : [])
            const mapped = items.map((r) => ({
              id: r.id,
              name: r.name,
              type: r.type,
              resourceGroup: r.resourceGroup || (r.id?.split('/')?.[4] ?? ''),
              location: r.location,
              tags: r.tags || {},
              status: r.healthStatus || 'Unknown',
              compliance: r.complianceStatus || 'Unknown',
              monthlyCost: typeof r.costEstimate === 'number' ? Number(r.costEstimate) : 0,
              cost: typeof r.costEstimate === 'number' ? Number(r.costEstimate) / 720 : 0,
              savings: 0,
              cpu: r.utilization?.cpu ?? 0,
              memory: r.utilization?.memory ?? 0,
              storage: r.utilization?.disk ?? 0,
              createdDate: r.createdDate || undefined,
              lastModified: r.lastModified || undefined,
              recommendations: r.recommendations || [],
            })) as AzureResource[]
            setResources(mapped)
            setIsUsingFallback(false)
            return
          }
        } catch (_) {
          // fall through to basic
        }

        // Fallback to basic resources endpoint
        setIsUsingFallback(true)
        const resp2 = await (authenticatedFetch as any)(`${API_BASE}/api/v1/resources`, { cache: 'no-store' })
        const data2 = await resp2.json()
        const raw2 = (data2 && (data2.resources ?? data2)) as any
        const items2: any[] = Array.isArray(raw2) ? raw2 : (raw2 ? [raw2] : [])
        setResources(items2 as AzureResource[])
      } catch (err: any) {
        setError(err?.message || 'Failed to load resources')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])
  
  return { resources, loading, error, isUsingFallback }
}

// Hook to fetch Azure policies
export function useAzurePolicies() {
  const [policies, setPolicies] = useState<AzurePolicy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const authenticatedFetch = useAuthenticatedFetch()
  
  useEffect(() => {
    fetchAzurePolicies(authenticatedFetch as any)
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
  const authenticatedFetch = useAuthenticatedFetch()
  
  useEffect(() => {
    fetchCostBreakdown(authenticatedFetch as any)
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
  const authenticatedFetch = useAuthenticatedFetch()
  
  useEffect(() => {
    fetchRbacAssignments(authenticatedFetch as any)
      .then(setAssignments)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])
  
  return { assignments, loading, error }
}