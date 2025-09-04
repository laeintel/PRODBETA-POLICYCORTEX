// PCG Platform Type Definitions

export interface PredictionData {
  id: string;
  type: 'compliance' | 'risk' | 'cost';
  score: number;
  confidence: number;
  prediction: string;
  recommendations: string[];
  impact: 'low' | 'medium' | 'high';
  timestamp: string;
  metadata?: {
    model: string;
    version: string;
    features: Record<string, any>;
  };
}

export interface EvidenceItem {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  verifiedBy: string;
  hash: string;
  status: 'pending' | 'verified' | 'rejected';
  chain?: {
    previousHash: string;
    blockNumber: number;
    signature: string;
  };
  attachments?: {
    name: string;
    url: string;
    size: number;
    mimeType: string;
  }[];
}

export interface ROIMetrics {
  totalSavings: number;
  preventedIncidents: number;
  complianceScore: number;
  riskReduction: number;
  automationHours: number;
  costAvoidance: number;
  breakdown?: {
    category: string;
    value: number;
    percentage: number;
  }[];
  trends?: {
    period: string;
    value: number;
  }[];
}

export interface PCGDashboardData {
  predictions: PredictionData[];
  evidence: EvidenceItem[];
  roi: ROIMetrics;
  summary: {
    activePredictions: number;
    verifiedEvidence: number;
    totalROI: number;
    complianceRate: number;
  };
}

export interface PCGAlert {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  source: 'prevent' | 'prove' | 'payback';
  actionRequired: boolean;
  actions?: {
    label: string;
    action: string;
  }[];
}

export interface PCGReport {
  id: string;
  name: string;
  type: 'compliance' | 'roi' | 'evidence' | 'executive';
  generatedAt: string;
  format: 'pdf' | 'csv' | 'json';
  url: string;
  size: number;
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrevious: boolean;
}