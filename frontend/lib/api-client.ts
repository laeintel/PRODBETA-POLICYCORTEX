// Simplified API Client for PolicyCortex PCG Platform

interface PredictionData {
  id: string;
  type: 'compliance' | 'risk' | 'cost';
  score: number;
  confidence: number;
  prediction: string;
  recommendations: string[];
  impact: 'low' | 'medium' | 'high';
  timestamp: string;
}

interface EvidenceItem {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  verifiedBy: string;
  hash: string;
  status: 'pending' | 'verified' | 'rejected';
}

interface ROIMetrics {
  totalSavings: number;
  preventedIncidents: number;
  complianceScore: number;
  riskReduction: number;
  automationHours: number;
  costAvoidance: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

class PCGApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async fetchWithRetry(url: string, options?: RequestInit): Promise<Response> {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      console.error('API Client Error:', error);
      throw error;
    }
  }

  // Prevent - Predictive Compliance APIs
  async getPredictions(): Promise<PredictionData[]> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/predictions`);
    return response.json();
  }

  async getPredictionById(id: string): Promise<PredictionData> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/predictions/${id}`);
    return response.json();
  }

  async createPrediction(data: Partial<PredictionData>): Promise<PredictionData> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/predictions`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
    return response.json();
  }

  // Prove - Evidence Chain APIs
  async getEvidence(): Promise<EvidenceItem[]> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/evidence`);
    return response.json();
  }

  async getEvidenceById(id: string): Promise<EvidenceItem> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/evidence/${id}`);
    return response.json();
  }

  async verifyEvidence(id: string): Promise<EvidenceItem> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/evidence/${id}/verify`, {
      method: 'POST',
    });
    return response.json();
  }

  async createEvidence(data: Partial<EvidenceItem>): Promise<EvidenceItem> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/evidence`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
    return response.json();
  }

  // Payback - ROI APIs
  async getROIMetrics(): Promise<ROIMetrics> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/roi/metrics`);
    return response.json();
  }

  async calculateROI(params: { startDate: string; endDate: string }): Promise<ROIMetrics> {
    const queryString = new URLSearchParams(params).toString();
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/roi/calculate?${queryString}`);
    return response.json();
  }

  async exportROIReport(format: 'pdf' | 'csv' | 'json' = 'pdf'): Promise<Blob> {
    const response = await this.fetchWithRetry(`${this.baseUrl}/api/v1/roi/export?format=${format}`);
    return response.blob();
  }

  // Dashboard APIs
  async getDashboardMetrics(): Promise<{
    predictions: PredictionData[];
    evidence: EvidenceItem[];
    roi: ROIMetrics;
  }> {
    const [predictions, evidence, roi] = await Promise.all([
      this.getPredictions(),
      this.getEvidence(),
      this.getROIMetrics(),
    ]);

    return { predictions, evidence, roi };
  }
}

// Export singleton instance
const apiClient = new PCGApiClient();
export default apiClient;

// Export types for use in components
export type { PredictionData, EvidenceItem, ROIMetrics };