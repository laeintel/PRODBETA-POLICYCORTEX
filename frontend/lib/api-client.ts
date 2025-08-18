// API Client for PolicyCortex Backend Services

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';
const GRAPHQL_URL = process.env.NEXT_PUBLIC_GRAPHQL_URL || 'http://localhost:4000/graphql';

interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

class PolicyCortexAPI {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor() {
    this.baseUrl = API_BASE_URL;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          ...this.headers,
          ...options.headers,
        },
      });

      const data = response.ok ? await response.json() : null;

      return {
        data,
        error: response.ok ? undefined : `Error: ${response.statusText}`,
        status: response.status,
      };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
        status: 0,
      };
    }
  }

  // Compliance APIs
  async getComplianceStatus() {
    return this.request<any>('/api/v1/compliance');
  }

  async runComplianceScan() {
    return this.request<any>('/api/v1/compliance/scan', {
      method: 'POST',
    });
  }

  async getComplianceViolations() {
    return this.request<any>('/api/v1/compliance/violations');
  }

  // Security APIs
  async getSecurityThreats() {
    return this.request<any>('/api/v1/security/threats');
  }

  async mitigateThreat(threatId: string) {
    return this.request<any>(`/api/v1/security/threats/${threatId}/mitigate`, {
      method: 'POST',
    });
  }

  async getSecurityAlerts() {
    return this.request<any>('/api/v1/security/alerts');
  }

  // Resource APIs
  async getResources(filters?: any) {
    const params = new URLSearchParams(filters).toString();
    return this.request<any>(`/api/v1/resources${params ? `?${params}` : ''}`);
  }

  async getResourceDetails(resourceId: string) {
    return this.request<any>(`/api/v1/resources/${resourceId}`);
  }

  async getResourceMetrics(resourceId: string) {
    return this.request<any>(`/api/v1/resources/${resourceId}/metrics`);
  }

  // Cost APIs
  async getCostAnalysis() {
    return this.request<any>('/api/v1/cost/analysis');
  }

  async getCostForecast() {
    return this.request<any>('/api/v1/cost/forecast');
  }

  async getCostRecommendations() {
    return this.request<any>('/api/v1/cost/recommendations');
  }

  // AI/ML APIs
  async getAIPredictions() {
    return this.request<any>('/api/v1/predictions');
  }

  async getCorrelations() {
    return this.request<any>('/api/v1/correlations');
  }

  async askAI(query: string) {
    return this.request<any>('/api/v1/conversation', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  // Metrics APIs
  async getUnifiedMetrics() {
    return this.request<any>('/api/v1/metrics');
  }

  async getHealthStatus() {
    return this.request<any>('/health');
  }

  // Policy APIs
  async getPolicies() {
    return this.request<any>('/api/v1/policies');
  }

  async getPolicyDetails(policyId: string) {
    return this.request<any>(`/api/v1/policies/${policyId}`);
  }

  async enforcePolicies() {
    return this.request<any>('/api/v1/policies/enforce', {
      method: 'POST',
    });
  }

  // WebSocket Connection for Real-time Updates
  connectWebSocket(onMessage: (data: any) => void) {
    const ws = new WebSocket(`ws://localhost:8080/ws`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => this.connectWebSocket(onMessage), 5000);
    };
    
    return ws;
  }
}

// Export singleton instance
export const api = new PolicyCortexAPI();

// React Query hooks for data fetching
export function useComplianceData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getComplianceStatus().then((response) => {
      if (response.error) {
        setError(response.error);
      } else {
        setData(response.data);
      }
      setLoading(false);
    });
  }, []);

  return { data, loading, error };
}

export function useSecurityThreats() {
  const [threats, setThreats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSecurityThreats().then((response) => {
      if (response.error) {
        setError(response.error);
      } else {
        setThreats(response.data?.threats || []);
      }
      setLoading(false);
    });
  }, []);

  return { threats, loading, error };
}

export function useResourceMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getUnifiedMetrics().then((response) => {
      if (response.error) {
        setError(response.error);
      } else {
        setMetrics(response.data);
      }
      setLoading(false);
    });
  }, []);

  return { metrics, loading, error };
}

export function useRealTimeUpdates(onUpdate: (data: any) => void) {
  useEffect(() => {
    const ws = api.connectWebSocket(onUpdate);
    
    return () => {
      ws.close();
    };
  }, [onUpdate]);
}