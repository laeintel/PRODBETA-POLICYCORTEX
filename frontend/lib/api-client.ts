// API Client for PolicyCortex - Connects to Live Azure Data

import type {
  DashboardMetrics,
  Alert,
  Activity,
  ComplianceStatus,
  PolicyViolation,
  RiskAssessment,
  CostSummary,
  Policy,
  IAMUser,
  RBACRole,
  PIMRequest,
  ConditionalAccessPolicy,
  ZeroTrustStatus,
  Entitlement,
  AccessReview,
  Resource,
  MonitoringMetric,
  AutomationWorkflow,
  Notification,
  Pipeline,
  Release,
  Artifact,
  Deployment,
  Build,
  Repository,
  PredictiveCompliance,
  Correlation,
  ChatResponse,
  UnifiedMetrics,
  AzureHealthStatus,
  ApiResponse,
  PaginatedResponse
} from '../types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

class ApiClient {
  private baseUrl: string;
  private cache: Map<string, CacheEntry<unknown>>;
  private cacheTimeout: number = 30000; // 30 seconds
  private csrfToken: string | null = null;

  constructor() {
    this.baseUrl = API_BASE_URL;
    this.cache = new Map();
    this.initializeCsrf();
  }

  private async initializeCsrf(): Promise<void> {
    if (typeof window === 'undefined') return;
    try {
      const response = await fetch('/api/auth/csrf');
      const data = await response.json();
      this.csrfToken = data.csrfToken;
    } catch (error) {
      console.warn('Failed to fetch CSRF token:', error);
    }
  }

  private getAuthHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    // CSRF token for state-changing operations
    if (this.csrfToken) {
      headers['X-CSRF-Token'] = this.csrfToken;
    }
    
    return headers;
  }

  private async fetchWithCache<T>(url: string): Promise<T> {
    const cached = this.cache.get(url) as CacheEntry<T> | undefined;
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    try {
      const response = await fetch(`${this.baseUrl}${url}`, {
        method: 'GET',
        headers: this.getAuthHeaders(),
        credentials: 'include', // Include cookies for authentication
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data: T = await response.json();
      this.cache.set(url, { data, timestamp: Date.now() });
      return data;
    } catch (error) {
      console.error(`Failed to fetch ${url}:`, error);
      // Return cached data if available, even if expired
      if (cached) {
        return cached.data;
      }
      throw error;
    }
  }

  private async post<T, R>(url: string, body: T): Promise<R> {
    const response = await fetch(`${this.baseUrl}${url}`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      credentials: 'include',
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    return response.json();
  }

  // Dashboard APIs
  async getDashboardMetrics(): Promise<DashboardMetrics> {
    return this.fetchWithCache<DashboardMetrics>('/api/v1/dashboard/metrics');
  }

  async getDashboardAlerts(): Promise<Alert[]> {
    return this.fetchWithCache<Alert[]>('/api/v1/dashboard/alerts');
  }

  async getDashboardActivities(): Promise<Activity[]> {
    return this.fetchWithCache<Activity[]>('/api/v1/dashboard/activities');
  }

  // Governance APIs
  async getComplianceStatus(): Promise<ComplianceStatus> {
    return this.fetchWithCache<ComplianceStatus>('/api/v1/governance/compliance/status');
  }

  async getComplianceViolations(): Promise<PolicyViolation[]> {
    return this.fetchWithCache<PolicyViolation[]>('/api/v1/governance/compliance/violations');
  }

  async getRiskAssessment(): Promise<RiskAssessment> {
    return this.fetchWithCache<RiskAssessment>('/api/v1/governance/risk/assessment');
  }

  async getCostSummary(): Promise<CostSummary> {
    return this.fetchWithCache<CostSummary>('/api/v1/governance/cost/summary');
  }

  async getPolicies(): Promise<Policy[]> {
    return this.fetchWithCache<Policy[]>('/api/v1/governance/policies');
  }

  // Security APIs
  async getIAMUsers(): Promise<IAMUser[]> {
    return this.fetchWithCache<IAMUser[]>('/api/v1/security/iam/users');
  }

  async getRBACRoles(): Promise<RBACRole[]> {
    return this.fetchWithCache<RBACRole[]>('/api/v1/security/rbac/roles');
  }

  async getPIMRequests(): Promise<PIMRequest[]> {
    return this.fetchWithCache<PIMRequest[]>('/api/v1/security/pim/requests');
  }

  async getConditionalAccessPolicies(): Promise<ConditionalAccessPolicy[]> {
    return this.fetchWithCache<ConditionalAccessPolicy[]>('/api/v1/security/conditional-access/policies');
  }

  async getZeroTrustStatus(): Promise<ZeroTrustStatus> {
    return this.fetchWithCache<ZeroTrustStatus>('/api/v1/security/zero-trust/status');
  }

  async getEntitlements(): Promise<Entitlement[]> {
    return this.fetchWithCache<Entitlement[]>('/api/v1/security/entitlements');
  }

  async getAccessReviews(): Promise<AccessReview[]> {
    return this.fetchWithCache<AccessReview[]>('/api/v1/security/access-reviews');
  }

  // Operations APIs
  async getResources(): Promise<Resource[]> {
    return this.fetchWithCache<Resource[]>('/api/v1/operations/resources');
  }

  async getMonitoringMetrics(): Promise<MonitoringMetric[]> {
    return this.fetchWithCache<MonitoringMetric[]>('/api/v1/operations/monitoring/metrics');
  }

  async getAutomationWorkflows(): Promise<AutomationWorkflow[]> {
    return this.fetchWithCache<AutomationWorkflow[]>('/api/v1/operations/automation/workflows');
  }

  async getNotifications(): Promise<Notification[]> {
    return this.fetchWithCache<Notification[]>('/api/v1/operations/notifications');
  }

  async getAlerts(): Promise<Alert[]> {
    return this.fetchWithCache<Alert[]>('/api/v1/operations/alerts');
  }

  // DevOps APIs
  async getPipelines(): Promise<Pipeline[]> {
    return this.fetchWithCache<Pipeline[]>('/api/v1/devops/pipelines');
  }

  async getReleases(): Promise<Release[]> {
    return this.fetchWithCache<Release[]>('/api/v1/devops/releases');
  }

  async getArtifacts(): Promise<Artifact[]> {
    return this.fetchWithCache<Artifact[]>('/api/v1/devops/artifacts');
  }

  async getDeployments(): Promise<Deployment[]> {
    return this.fetchWithCache<Deployment[]>('/api/v1/devops/deployments');
  }

  async getBuilds(): Promise<Build[]> {
    return this.fetchWithCache<Build[]>('/api/v1/devops/builds');
  }

  async getRepos(): Promise<Repository[]> {
    return this.fetchWithCache<Repository[]>('/api/v1/devops/repos');
  }

  // AI APIs
  async getPredictiveCompliance(): Promise<PredictiveCompliance> {
    return this.fetchWithCache<PredictiveCompliance>('/api/v1/ai/predictive/compliance');
  }

  async getCorrelations(): Promise<Correlation[]> {
    return this.fetchWithCache<Correlation[]>('/api/v1/ai/correlations');
  }

  async sendChatMessage(message: string): Promise<ChatResponse> {
    return this.post<{ message: string }, ChatResponse>('/api/v1/ai/chat', { message });
  }

  async getUnifiedMetrics(): Promise<UnifiedMetrics> {
    return this.fetchWithCache<UnifiedMetrics>('/api/v1/ai/unified/metrics');
  }

  // Health check
  async checkAzureHealth(): Promise<AzureHealthStatus> {
    return this.fetchWithCache<AzureHealthStatus>('/api/v1/health/azure');
  }

  // Clear cache
  clearCache() {
    this.cache.clear();
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export hooks for React components
export function useApiClient() {
  return apiClient;
}