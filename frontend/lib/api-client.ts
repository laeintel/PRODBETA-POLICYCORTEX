// API Client for PolicyCortex - Connects to Live Azure Data

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;
  private cache: Map<string, { data: any; timestamp: number }>;
  private cacheTimeout: number = 30000; // 30 seconds

  constructor() {
    this.baseUrl = API_BASE_URL;
    this.cache = new Map();
  }

  private async fetchWithCache(url: string): Promise<any> {
    const cached = this.cache.get(url);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    try {
      const response = await fetch(`${this.baseUrl}${url}`, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
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

  // Dashboard APIs
  async getDashboardMetrics() {
    return this.fetchWithCache('/api/v1/dashboard/metrics');
  }

  async getDashboardAlerts() {
    return this.fetchWithCache('/api/v1/dashboard/alerts');
  }

  async getDashboardActivities() {
    return this.fetchWithCache('/api/v1/dashboard/activities');
  }

  // Governance APIs
  async getComplianceStatus() {
    return this.fetchWithCache('/api/v1/governance/compliance/status');
  }

  async getComplianceViolations() {
    return this.fetchWithCache('/api/v1/governance/compliance/violations');
  }

  async getRiskAssessment() {
    return this.fetchWithCache('/api/v1/governance/risk/assessment');
  }

  async getCostSummary() {
    return this.fetchWithCache('/api/v1/governance/cost/summary');
  }

  async getPolicies() {
    return this.fetchWithCache('/api/v1/governance/policies');
  }

  // Security APIs
  async getIAMUsers() {
    return this.fetchWithCache('/api/v1/security/iam/users');
  }

  async getRBACRoles() {
    return this.fetchWithCache('/api/v1/security/rbac/roles');
  }

  async getPIMRequests() {
    return this.fetchWithCache('/api/v1/security/pim/requests');
  }

  async getConditionalAccessPolicies() {
    return this.fetchWithCache('/api/v1/security/conditional-access/policies');
  }

  async getZeroTrustStatus() {
    return this.fetchWithCache('/api/v1/security/zero-trust/status');
  }

  async getEntitlements() {
    return this.fetchWithCache('/api/v1/security/entitlements');
  }

  async getAccessReviews() {
    return this.fetchWithCache('/api/v1/security/access-reviews');
  }

  // Operations APIs
  async getResources() {
    return this.fetchWithCache('/api/v1/operations/resources');
  }

  async getMonitoringMetrics() {
    return this.fetchWithCache('/api/v1/operations/monitoring/metrics');
  }

  async getAutomationWorkflows() {
    return this.fetchWithCache('/api/v1/operations/automation/workflows');
  }

  async getNotifications() {
    return this.fetchWithCache('/api/v1/operations/notifications');
  }

  async getAlerts() {
    return this.fetchWithCache('/api/v1/operations/alerts');
  }

  // DevOps APIs
  async getPipelines() {
    return this.fetchWithCache('/api/v1/devops/pipelines');
  }

  async getReleases() {
    return this.fetchWithCache('/api/v1/devops/releases');
  }

  async getArtifacts() {
    return this.fetchWithCache('/api/v1/devops/artifacts');
  }

  async getDeployments() {
    return this.fetchWithCache('/api/v1/devops/deployments');
  }

  async getBuilds() {
    return this.fetchWithCache('/api/v1/devops/builds');
  }

  async getRepos() {
    return this.fetchWithCache('/api/v1/devops/repos');
  }

  // AI APIs
  async getPredictiveCompliance() {
    return this.fetchWithCache('/api/v1/ai/predictive/compliance');
  }

  async getCorrelations() {
    return this.fetchWithCache('/api/v1/ai/correlations');
  }

  async sendChatMessage(message: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/ai/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    return response.json();
  }

  async getUnifiedMetrics() {
    return this.fetchWithCache('/api/v1/ai/unified/metrics');
  }

  // Health check
  async checkAzureHealth() {
    return this.fetchWithCache('/api/v1/health/azure');
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