// PolicyCortex API Client - Connected to Patent-Based Architecture
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

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
  intent: string;
  confidence: number;
  suggested_actions: string[];
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

  // Patent 1: Unified AI Platform
  async getUnifiedMetrics(): Promise<GovernanceMetrics> {
    const response = await fetch(`${this.baseURL}/api/v1/metrics`);
    if (!response.ok) {
      throw new Error(`Failed to fetch metrics: ${response.statusText}`);
    }
    return response.json();
  }

  // Patent 2: Predictive Compliance
  async getPredictions(): Promise<any[]> {
    const response = await fetch(`${this.baseURL}/api/v1/predictions`);
    if (!response.ok) {
      throw new Error(`Failed to fetch predictions: ${response.statusText}`);
    }
    return response.json();
  }

  // Patent 3: Conversational Intelligence
  async processConversation(request: ConversationRequest): Promise<ConversationResponse> {
    const response = await fetch(`${this.baseURL}/api/v1/conversation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new Error(`Failed to process conversation: ${response.statusText}`);
    }
    return response.json();
  }

  // Patent 4: Cross-Domain Correlation
  async getCorrelations(): Promise<CrossDomainCorrelation[]> {
    const response = await fetch(`${this.baseURL}/api/v1/correlations`);
    if (!response.ok) {
      throw new Error(`Failed to fetch correlations: ${response.statusText}`);
    }
    return response.json();
  }

  // Proactive Recommendations
  async getRecommendations(): Promise<ProactiveRecommendation[]> {
    const response = await fetch(`${this.baseURL}/api/v1/recommendations`);
    if (!response.ok) {
      throw new Error(`Failed to fetch recommendations: ${response.statusText}`);
    }
    return response.json();
  }

  // Health Check
  async getHealth(): Promise<{ status: string; version: string; service: string; patents: string[] }> {
    const response = await fetch(`${this.baseURL}/health`);
    if (!response.ok) {
      throw new Error(`Failed to fetch health: ${response.statusText}`);
    }
    return response.json();
  }

  // Legacy endpoints for compatibility
  async getPolicies(): Promise<any[]> {
    const response = await fetch(`${this.baseURL}/api/v1/policies`);
    if (!response.ok) {
      throw new Error(`Failed to fetch policies: ${response.statusText}`);
    }
    return response.json();
  }

  async getResources(): Promise<any[]> {
    const response = await fetch(`${this.baseURL}/api/v1/resources`);
    if (!response.ok) {
      throw new Error(`Failed to fetch resources: ${response.statusText}`);
    }
    return response.json();
  }

  async getCompliance(): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/v1/compliance`);
    if (!response.ok) {
      throw new Error(`Failed to fetch compliance: ${response.statusText}`);
    }
    return response.json();
  }
}

// Export singleton instance
export const api = new PolicyCortexAPI();

// React Hook for data fetching
export function useGovernanceData() {
  const [metrics, setMetrics] = React.useState<GovernanceMetrics | null>(null);
  const [recommendations, setRecommendations] = React.useState<ProactiveRecommendation[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [metricsData, recommendationsData] = await Promise.all([
          api.getUnifiedMetrics(),
          api.getRecommendations(),
        ]);
        setMetrics(metricsData);
        setRecommendations(recommendationsData);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Only refresh on initial load, no auto-refresh to prevent constant page updates
    // Can be re-enabled later with longer intervals if needed
  }, []);

  return { metrics, recommendations, loading, error };
}

// React Hook for conversations
export function useConversation() {
  const [sessionId] = React.useState(() => `session-${Date.now()}`);
  const [loading, setLoading] = React.useState(false);

  const sendMessage = async (query: string, context?: ConversationRequest['context']) => {
    setLoading(true);
    try {
      const response = await api.processConversation({
        query,
        context,
        session_id: sessionId,
      });
      return response;
    } finally {
      setLoading(false);
    }
  };

  return { sendMessage, loading, sessionId };
}

import React from 'react';