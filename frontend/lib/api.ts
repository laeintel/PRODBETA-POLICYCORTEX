/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

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
      const { msalConfig, coreApiRequest, azureManagementRequest } = await import('./auth-config');
      
      const msalInstance = new PublicClientApplication(msalConfig);
      await msalInstance.initialize();
      const accounts = msalInstance.getAllAccounts();
      
      if (accounts.length > 0) {
        // Prefer core API scope if configured; fall back to Azure Management
        const request = coreApiRequest || azureManagementRequest;
        const tokenRequest = { ...request, account: accounts[0] };
        const tokenResponse = await msalInstance.acquireTokenSilent(tokenRequest);
        
        // Extract tenant ID from token claims
        const accountWithClaims = accounts[0] as {
          idTokenClaims?: { tid?: string }
        };
        const tenantId = accountWithClaims?.idTokenClaims?.tid
          || process.env?.NEXT_PUBLIC_AZURE_TENANT_ID
          || (typeof window !== 'undefined' ? window.localStorage.getItem('tenantId') || undefined : undefined);
        
        return {
          'Authorization': `Bearer ${tokenResponse.accessToken}`,
          ...(tenantId ? { 'X-Tenant-ID': tenantId } : {})
        };
      }
    } catch (error) {
      // Local/dev: proceed without auth header
      console.debug('No authentication available (dev):', error);
    }
    
    return {};
  }

  // Patent 1: Unified AI Platform - Hot cached for real-time governance
  async getUnifiedMetrics(): Promise<GovernanceMetrics> {
    const res = await performanceApi.get<GovernanceMetrics | { data: GovernanceMetrics }>('/api/v1/metrics', { cache: 'hot', ttl: 30000, headers: await this.getAuthHeaders() });
    // Accept either flat object or { data: {...} }
    if (res && typeof res === 'object' && 'data' in res) {
      return res.data as GovernanceMetrics;
    }
    return res as GovernanceMetrics;
  }

  // Patent 2: Predictive Compliance - Warm cached for frequent access
  async getPredictions(): Promise<Array<{
    resource_id: string;
    prediction_type: string;
    probability: number;
    timeframe: string;
    impact: string;
    recommended_actions: string[];
  }>> {
    return performanceApi.get('/api/v1/predictions', { cache: 'warm', ttl: 300000, headers: await this.getAuthHeaders() });
  }

  // Patent 3: Conversational Intelligence - No cache for real-time interaction
  async processConversation(request: ConversationRequest): Promise<ConversationResponse> {
    // Call backend conversation endpoint; fall back gracefully in dev
    let base: { response?: string; confidence?: number; suggestions?: string[] } = {};
    try {
      base = await performanceApi.post<{ response?: string; confidence?: number; suggestions?: string[] }>('/api/v1/conversation', request, {
        headers: await this.getAuthHeaders(),
        invalidateCache: ['conversation', 'recommendations']
      });
    } catch (err) {
      console.warn('Conversation API failed; using local fallback', err instanceof Error ? err.message : err);
      base = { response: 'Demo response: conversation unavailable locally.', confidence: 0.7, suggestions: ['open_dashboard'] };
    }

    const response: ConversationResponse = {
      response: base?.response ?? '',
      confidence: typeof base?.confidence === 'number' ? base.confidence * 100 : 0,
      suggested_actions: base?.suggestions ?? [],
    };

    // Simple intent inference
    const q = (request?.query || '').toLowerCase();
    if (q.includes('cost')) response.intent = 'cost_inquiry';
    else if (q.includes('security')) response.intent = 'security_insight';
    else if (q.includes('compliance') || q.includes('policy')) response.intent = 'compliance_policy';

    // Opportunistically generate a sample policy if user asks for policy
    if (q.includes('policy')) {
      try {
        const gen = await performanceApi.post<{ policy?: object }>('/api/v1/policies/generate', {
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
    const response = await performanceApi.get<ProactiveRecommendation[] | { recommendations: ProactiveRecommendation[] }>('/api/v1/recommendations', { cache: 'hot', ttl: 60000, headers: await this.getAuthHeaders() });
    if (Array.isArray(response)) {
      return response;
    }
    if (response && typeof response === 'object' && 'recommendations' in response) {
      return response.recommendations;
    }
    return [];
  }

  // Health Check - No cache for real-time status
  async getHealth(): Promise<{ status: string; version: string; service: string; patents: string[] }> {
    return performanceApi.get('/health', { cache: 'none', timeout: 5000 });
  }

  // Legacy endpoints for compatibility - Warm cached
  async getPolicies(): Promise<Array<{
    id: string;
    name: string;
    type: string;
    enabled: boolean;
    scope: string[];
  }>> {
    return performanceApi.get('/api/v1/policies', { 
      cache: 'warm', 
      ttl: 300000,
      headers: await this.getAuthHeaders()
    });
  }

  async getResources(): Promise<Array<{
    id: string;
    name: string;
    type: string;
    location: string;
    tags: Record<string, string>;
  }>> {
    return performanceApi.get('/api/v1/resources', { 
      cache: 'warm', 
      ttl: 300000,
      headers: await this.getAuthHeaders()
    });
  }

  async getCompliance(): Promise<{
    overall_score: number;
    framework: string;
    controls_passed: number;
    controls_failed: number;
    controls_total: number;
  }> {
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
    predictions: Array<{
      resource_id: string;
      prediction_type: string;
      probability: number;
      timeframe: string;
      impact: string;
      recommended_actions: string[];
    }>;
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
      policies: { total: 0, active: 0, violations: 0, automated: 0, compliance_rate: 0, prediction_accuracy: 0 },
      rbac: { users: 0, roles: 0, violations: 0, risk_score: 0, anomalies_detected: 0 },
      costs: { current_spend: 0, predicted_spend: 0, savings_identified: 0, optimization_rate: 0 },
      network: { endpoints: 0, active_threats: 0, blocked_attempts: 0, latency_ms: 0 },
      resources: { total: 0, optimized: 0, idle: 0, overprovisioned: 0 },
      ai: { accuracy: 0, predictions_made: 0, automations_executed: 0, learning_progress: 0 }
    } as GovernanceMetrics;

    const metricsPayload = results[0] instanceof Error ? defaultMetrics : results[0];
    return {
      metrics: (metricsPayload as { data?: GovernanceMetrics })?.data ?? (metricsPayload as GovernanceMetrics),
      recommendations: results[1] instanceof Error ? [] : (Array.isArray(results[1]) ? results[1] : (results[1] as { recommendations?: ProactiveRecommendation[] })?.recommendations || []),
      correlations: results[2] instanceof Error ? [] : results[2] as CrossDomainCorrelation[],
      predictions: results[3] instanceof Error ? [] : results[3] as Array<{
        resource_id: string;
        prediction_type: string;
        probability: number;
        timeframe: string;
        impact: string;
        recommended_actions: string[];
      }>
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
    predictions: Array<{
      resource_id: string;
      prediction_type: string;
      probability: number;
      timeframe: string;
      impact: string;
      recommended_actions: string[];
    }>;
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