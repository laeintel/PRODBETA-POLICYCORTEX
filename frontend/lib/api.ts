// PolicyCortex API Client - High-Performance with Caching and Connection Pooling
import { performanceApi, queryKeys } from './performance-api'
import React from 'react'

// Use relative URLs to leverage Next.js proxy configuration
const API_BASE_URL = '';

export interface GovernanceMetrics {
  policies: {
    total: number;
    active: number;
    violations: number;
    automated: number;
    compliance_rate: number;
    prediction_accuracy: number;
  };
  rbac: {
    users: number;
    roles: number;
    violations: number;
    risk_score: number;
    anomalies_detected: number;
  };
  costs: {
    current_spend: number;
    predicted_spend: number;
    savings_identified: number;
    optimization_rate: number;
  };
  network: {
    endpoints: number;
    active_threats: number;
    blocked_attempts: number;
    latency_ms: number;
  };
  resources: {
    total: number;
    optimized: number;
    idle: number;
    overprovisioned: number;
  };
  ai: {
    accuracy: number;
    predictions_made: number;
    automations_executed: number;
    learning_progress: number;
  };
}

export interface ProactiveRecommendation {
  id: string;
  recommendation_type: string;
  severity: string;
  title: string;
  description: string;
  potential_savings?: number;
  risk_reduction?: number;
  automation_available: boolean;
  confidence: number;
}

export interface ConversationRequest {
  query: string;
  context?: {
    previous_intents: string[];
    entities: Array<{
      entity_type: string;
      value: string;
      confidence: number;
    }>;
    turn_count: number;
  };
  session_id: string;
}

export interface ConversationResponse {
  response: string;
  intent?: string;
  confidence: number;
  suggested_actions?: string[];
  generated_policy?: string;
}

export interface CrossDomainCorrelation {
  correlation_id: string;
  domains: string[];
  correlation_strength: number;
  causal_relationship?: {
    source_domain: string;
    target_domain: string;
    lag_time_hours: number;
    confidence: number;
  };
  impact_predictions: Array<{
    domain: string;
    metric: string;
    predicted_change: number;
    time_to_impact_hours: number;
  }>;
}

// API Client Class
class PolicyCortexAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async getAuthHeaders(): Promise<Record<string, string>> {
    // Skip auth headers during SSR
    if (typeof window === 'undefined') {
      return {};
    }
    
    try {
      // If we have MSAL context, get the access token
      const { PublicClientApplication } = await import('@azure/msal-browser');
      const { msalConfig } = await import('./auth-config');
      
      const msalInstance = new PublicClientApplication(msalConfig);
      await msalInstance.initialize();
      const accounts = msalInstance.getAllAccounts();
      
      if (accounts.length > 0) {
        const tokenResponse = await msalInstance.acquireTokenSilent({
          scopes: ['https://management.azure.com/user_impersonation'],
          account: accounts[0]
        });
        const tenantId = (accounts[0] as any)?.idTokenClaims?.tid as string | undefined
          || (typeof process !== 'undefined' ? (process as any).env?.NEXT_PUBLIC_AZURE_TENANT_ID : undefined)
          || (typeof window !== 'undefined' ? window.localStorage.getItem('tenantId') || undefined : undefined);
        
        return {
          'Authorization': `Bearer ${tokenResponse.accessToken}`,
          ...(tenantId ? { 'X-Tenant-ID': tenantId } : {})
        };
      }
    } catch (error) {
      console.debug('No authentication available:', error);
    }
    
    return {};
  }

  // Patent 1: Unified AI Platform - Hot cached for real-time governance
  async getUnifiedMetrics(): Promise<GovernanceMetrics> {
    return performanceApi.get('/api/v1/metrics', { cache: 'hot', ttl: 30000, headers: await this.getAuthHeaders() });
  }

  // Patent 2: Predictive Compliance - Warm cached for frequent access
  async getPredictions(): Promise<any[]> {
    return performanceApi.get('/api/v1/predictions', { cache: 'warm', ttl: 300000, headers: await this.getAuthHeaders() });
  }

  // Patent 3: Conversational Intelligence - No cache for real-time interaction
  async processConversation(request: ConversationRequest): Promise<ConversationResponse> {
    // Call chat endpoint (returns model + suggestions)
    const base = await performanceApi.post<any>('/api/v1/chat', request, {
      headers: await this.getAuthHeaders(),
      invalidateCache: ['conversation', 'recommendations']
    });

    const response: ConversationResponse = {
      response: base?.response ?? '',
      confidence: typeof base?.confidence === 'number' ? base.confidence * 100 : 0,
      suggested_actions: base?.suggestions ?? [],
    };

    // Simple intent inference
    const q = request.query.toLowerCase();
    if (q.includes('cost')) response.intent = 'cost_inquiry';
    else if (q.includes('security')) response.intent = 'security_insight';
    else if (q.includes('compliance') || q.includes('policy')) response.intent = 'compliance_policy';

    // Opportunistically generate a sample policy if user asks for policy
    if (q.includes('policy')) {
      try {
        const gen = await performanceApi.post<any>('/api/v1/policies/generate', {
          requirement: request.query,
          provider: 'azure',
          framework: undefined,
        }, { headers: await this.getAuthHeaders() });
        if (gen?.policy) {
          response.generated_policy = JSON.stringify(gen.policy);
        }
      } catch (_) {
        // ignore
      }
    }

    return response;
  }

  // Patent 4: Cross-Domain Correlation - Warm cached
  async getCorrelations(): Promise<CrossDomainCorrelation[]> {
    return performanceApi.get('/api/v1/correlations', { cache: 'warm', ttl: 300000, headers: await this.getAuthHeaders() });
  }

  // Proactive Recommendations - Hot cached for immediate actions
  async getRecommendations(): Promise<ProactiveRecommendation[]> {
    const response = await performanceApi.get('/api/v1/recommendations', { cache: 'hot', ttl: 60000, headers: await this.getAuthHeaders() });
    return response?.recommendations || [];
  }

  // Health Check - No cache for real-time status
  async getHealth(): Promise<{ status: string; version: string; service: string; patents: string[] }> {
    return performanceApi.get('/health', { cache: 'none', timeout: 5000 });
  }

  // Legacy endpoints for compatibility - Warm cached
  async getPolicies(): Promise<any[]> {
    return performanceApi.get('/api/v1/policies', { 
      cache: 'warm', 
      ttl: 300000,
      headers: await this.getAuthHeaders()
    });
  }

  async getResources(): Promise<any[]> {
    return performanceApi.get('/api/v1/resources', { 
      cache: 'warm', 
      ttl: 300000,
      headers: await this.getAuthHeaders()
    });
  }

  async getCompliance(): Promise<any> {
    return performanceApi.get('/api/v1/compliance', { 
      cache: 'hot', 
      ttl: 60000,
      headers: await this.getAuthHeaders()
    });
  }

  // Batch loading for dashboard performance
  async getDashboardData(): Promise<{
    metrics: GovernanceMetrics | null;
    recommendations: ProactiveRecommendation[];
    correlations: CrossDomainCorrelation[];
    predictions: any[];
  }> {
    const headers = await this.getAuthHeaders();
    const requests = [
      { endpoint: '/api/v1/metrics', options: { cache: 'hot', ttl: 30000, headers } },
      { endpoint: '/api/v1/recommendations', options: { cache: 'hot', ttl: 60000, headers } },
      { endpoint: '/api/v1/correlations', options: { cache: 'warm', ttl: 300000, headers } },
      { endpoint: '/api/v1/predictions', options: { cache: 'warm', ttl: 300000, headers } }
    ];

    const results = await performanceApi.batch(requests);
    
    // Handle any errors in the batch
    const errors = results.filter(r => r instanceof Error);
    if (errors.length > 0) {
      console.warn('Some dashboard requests failed:', errors);
    }

    const defaultMetrics: GovernanceMetrics = {
      policies: { total: 0, active: 0, violations: 0, automated: 0 as any, compliance_rate: 0, prediction_accuracy: 0 },
      rbac: { users: 0, roles: 0, violations: 0, risk_score: 0, anomalies_detected: 0 },
      costs: { current_spend: 0, predicted_spend: 0, savings_identified: 0, optimization_rate: 0 },
      network: { endpoints: 0, active_threats: 0, blocked_attempts: 0, latency_ms: 0 },
      resources: { total: 0, optimized: 0, idle: 0, overprovisioned: 0 },
      ai: { accuracy: 0, predictions_made: 0, automations_executed: 0, learning_progress: 0 }
    } as unknown as GovernanceMetrics;

    return {
      metrics: results[0] instanceof Error ? defaultMetrics : (results[0] as GovernanceMetrics),
      recommendations: results[1] instanceof Error ? [] : ((results[1] as any)?.recommendations || results[1] || []),
      correlations: results[2] instanceof Error ? [] : results[2] as CrossDomainCorrelation[],
      predictions: results[3] instanceof Error ? [] : results[3] as any[]
    };
  }

  // Cache management
  invalidateCache(pattern: string) {
    performanceApi.invalidateCache(pattern);
  }

  // Performance stats for monitoring
  getPerformanceStats() {
    return performanceApi.getStats();
  }
}

// Export singleton instance
export const api = new PolicyCortexAPI();

// High-performance React Hook for governance data with intelligent batching
export function useGovernanceData() {
  const [data, setData] = React.useState<{
    metrics: GovernanceMetrics | null;
    recommendations: ProactiveRecommendation[];
    correlations: CrossDomainCorrelation[];
    predictions: any[];
  }>({
    metrics: null,
    recommendations: [],
    correlations: [],
    predictions: []
  });
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = React.useState<Date | null>(null);

  React.useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Use batch loading for maximum performance
        const dashboardData = await api.getDashboardData();
        
        setData(dashboardData);
        setError(null);
        setLastUpdate(new Date());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Smart refresh: only for critical governance data
    const refreshInterval = setInterval(async () => {
      try {
        // Only refresh metrics and recommendations (hot data)
        const [metrics, recommendations] = await Promise.all([
          api.getUnifiedMetrics(),
          api.getRecommendations()
        ]);
        
        setData(prev => ({ ...prev, metrics, recommendations }));
        setLastUpdate(new Date());
      } catch (err) {
        console.debug('Background refresh failed:', err);
      }
    }, 30000); // 30 seconds for hot data

    return () => clearInterval(refreshInterval);
  }, []);

  return { 
    metrics: data.metrics, 
    recommendations: data.recommendations,
    correlations: data.correlations,
    predictions: data.predictions,
    loading, 
    error,
    lastUpdate,
    // Cache management
    invalidateCache: (pattern: string) => api.invalidateCache(pattern),
    performanceStats: () => api.getPerformanceStats()
  };
}

// High-performance React Hook for conversations with context management
export function useConversation() {
  const [sessionId] = React.useState(() => `session-${Date.now()}`);
  const [loading, setLoading] = React.useState(false);
  const [conversationHistory, setConversationHistory] = React.useState<Array<{
    query: string;
    response: ConversationResponse;
    timestamp: Date;
  }>>([]);

  const sendMessage = async (query: string, context?: ConversationRequest['context']) => {
    setLoading(true);
    try {
      // Build context from conversation history
        const enhancedContext = {
          previous_intents: conversationHistory.slice(-3).map(h => h.response.intent || '').filter(Boolean) as string[],
        entities: context?.entities || [],
        turn_count: conversationHistory.length,
        ...context
      };

      const response = await api.processConversation({
        query,
        context: enhancedContext,
        session_id: sessionId,
      });

      // Update conversation history
      setConversationHistory(prev => [...prev, {
        query,
        response,
        timestamp: new Date()
      }].slice(-10)); // Keep last 10 interactions

      return response;
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setConversationHistory([]);
  };

  return { 
    sendMessage, 
    loading, 
    sessionId, 
    conversationHistory,
    clearHistory
  };
}