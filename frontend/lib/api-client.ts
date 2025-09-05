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
    const data = await response.json();
    
    // Handle PCG server response format which has predictions array inside
    if (data.predictions && Array.isArray(data.predictions)) {
      // Transform PCG format to frontend format
      return data.predictions.map((p: any) => ({
        id: p.id,
        type: p.violation_type === 'data_encryption' || p.violation_type === 'excessive_permissions' ? 'compliance' : 
              p.violation_type === 'unpatched_system' ? 'risk' : 'cost',
        score: Math.round(p.probability * 100),
        confidence: Math.round(p.probability * 100),
        prediction: `${p.resource_name}: ${p.recommended_action}`,
        recommendations: [p.recommended_action],
        impact: p.severity.toLowerCase() as 'low' | 'medium' | 'high',
        timestamp: p.prediction_date
      }));
    }
    
    // Fallback for direct array response
    return Array.isArray(data) ? data : [];
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
    const data = await response.json();
    
    // Handle PCG server response format which has evidence_items array inside
    if (data.evidence_items && Array.isArray(data.evidence_items)) {
      // Transform PCG format to frontend format
      return data.evidence_items.map((e: any) => ({
        id: e.id,
        type: e.control || 'compliance',
        description: e.details || `Control ${e.control} - ${e.status}`,
        timestamp: e.timestamp,
        verifiedBy: e.verified ? 'system' : 'pending',
        hash: e.hash,
        status: e.verified ? 'verified' : 'pending'
      }));
    }
    
    // Fallback for direct array response
    return Array.isArray(data) ? data : [];
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
    const data = await response.json();
    
    // Handle PCG server response format
    if (data.summary) {
      // Transform PCG format to frontend format
      return {
        totalSavings: data.summary.total_savings || 0,
        preventedIncidents: data.breakdown?.prevented_incidents?.count || 0,
        complianceScore: Math.round((data.summary.roi_percentage || 0) / 4), // Convert ROI to score
        riskReduction: Math.round(data.summary.cloud_cost_reduction * 100) || 0,
        automationHours: data.breakdown?.automated_remediation?.hours_saved || 0,
        costAvoidance: data.breakdown?.compliance_penalties_avoided?.value || 0
      };
    }
    
    // Fallback for direct response
    return data;
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