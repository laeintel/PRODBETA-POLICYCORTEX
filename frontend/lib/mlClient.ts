/**
 * ML Client for Patent #4 Predictive Policy Compliance Engine
 * Connects frontend to real ML endpoints and WebSocket server
 */

import { io, Socket } from 'socket.io-client';

// API Configuration
const ML_API_BASE = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8080/api/v1';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8765';

// Types
export interface PredictionRequest {
  resourceId: string;
  tenantId: string;
  configuration: any;
  timeSeriesData?: TimeSeriesPoint[];
  policyContext?: PolicyContext;
  priority?: number;
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  metricName: string;
}

export interface PolicyContext {
  policyId: string;
  policyType: string;
  attachments: string[];
  inheritanceDepth: number;
}

export interface PredictionResponse {
  predictionId: string;
  resourceId: string;
  violationProbability: number;
  timeToViolationHours?: number;
  confidenceScore: number;
  confidenceInterval: [number, number];
  riskLevel: 'critical' | 'high' | 'medium' | 'low';
  recommendations: string[];
  inferenceTimeMs: number;
  modelVersion: string;
  timestamp: string;
}

export interface ViolationForecast {
  resourceId: string;
  policyId: string;
  forecastWindowHours: number;
  violationProbability: number;
  confidence: number;
  predictedTime: string;
}

export interface RiskAssessment {
  resourceId: string;
  riskScore: number;
  riskLevel: string;
  impactFactors: {
    security: number;
    compliance: number;
    operational: number;
    financial: number;
  };
  uncertaintySources: {
    epistemic: number;
    aleatoric: number;
    model: number;
    calibration: number;
  };
  recommendations: string[];
}

export interface FeatureImportance {
  featureImportance: Array<{
    featureName: string;
    importanceScore: number;
    contributionDirection: 'positive' | 'negative';
  }>;
  shapValues?: number[];
  visualizationData?: any;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  inferenceTimeP50Ms: number;
  inferenceTimeP95Ms: number;
  inferenceTimeP99Ms: number;
  meetsPatentRequirements: boolean;
}

export interface DriftMetrics {
  driftDetected: boolean;
  driftScore: number;
  driftVelocity: number;
  timeToViolationHours?: number;
  confidence: number;
  recommendations: string[];
}

// ML API Client
export class MLApiClient {
  private authToken: string | null = null;

  constructor(authToken?: string) {
    this.authToken = authToken || null;
  }

  private async fetch(endpoint: string, options: RequestInit = {}) {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers as Record<string, string>,
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    const response = await fetch(`${ML_API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`ML API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Prediction APIs
  async getAllPredictions(): Promise<PredictionResponse[]> {
    return this.fetch('/predictions');
  }

  async createPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    return this.fetch('/predictions', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getViolationForecasts(
    limit?: number,
    timeWindowHours?: number,
    minProbability?: number
  ): Promise<ViolationForecast[]> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (timeWindowHours) params.append('time_window_hours', timeWindowHours.toString());
    if (minProbability) params.append('min_probability', minProbability.toString());

    return this.fetch(`/predictions/violations?${params}`);
  }

  async getRiskAssessment(resourceId: string): Promise<RiskAssessment> {
    return this.fetch(`/predictions/risk-score/${resourceId}`);
  }

  async triggerRemediation(
    predictionId: string,
    autoRemediate: boolean = false,
    dryRun: boolean = true
  ) {
    return this.fetch(`/predictions/remediate/${predictionId}`, {
      method: 'POST',
      body: JSON.stringify({
        auto_remediate: autoRemediate,
        dry_run: dryRun,
        approval_required: !autoRemediate,
      }),
    });
  }

  // Model Management APIs
  async getFeatureImportance(
    modelName: string = 'ensemble',
    predictionId?: string,
    globalAnalysis: boolean = true
  ): Promise<FeatureImportance> {
    return this.fetch('/ml/feature-importance', {
      method: 'GET',
      body: JSON.stringify({
        model_name: modelName,
        prediction_id: predictionId,
        global_analysis: globalAnalysis,
      }),
    });
  }

  async triggerRetraining(
    triggerReason: string,
    useLatestData: boolean = true,
    hyperparameterTuning: boolean = false
  ) {
    return this.fetch('/ml/retrain', {
      method: 'POST',
      body: JSON.stringify({
        trigger_reason: triggerReason,
        use_latest_data: useLatestData,
        hyperparameter_tuning: hyperparameterTuning,
        validation_split: 0.2,
      }),
    });
  }

  async getModelMetrics(): Promise<ModelMetrics> {
    return this.fetch('/ml/metrics');
  }

  async submitFeedback(
    predictionId: string,
    feedbackType: 'correct' | 'incorrect' | 'false_positive' | 'false_negative',
    correctLabel?: boolean,
    accuracyRating?: number,
    comments?: string,
    userId: string = 'user'
  ) {
    return this.fetch('/ml/feedback', {
      method: 'POST',
      body: JSON.stringify({
        prediction_id: predictionId,
        feedback_type: feedbackType,
        correct_label: correctLabel,
        accuracy_rating: accuracyRating,
        comments,
        user_id: userId,
      }),
    });
  }

  // Configuration APIs
  async getResourceConfiguration(resourceId: string) {
    return this.fetch(`/configurations/${resourceId}`);
  }

  async analyzeDrift(resourceId: string, configuration: any): Promise<DriftMetrics> {
    return this.fetch('/configurations/drift-analysis', {
      method: 'POST',
      body: JSON.stringify({
        resource_id: resourceId,
        configuration,
      }),
    });
  }

  async getBaselineConfiguration(resourceId: string) {
    return this.fetch(`/configurations/baseline/${resourceId}`);
  }

  // Explainability APIs
  async getPredictionExplanation(predictionId: string) {
    return this.fetch(`/explanations/${predictionId}`);
  }

  async getGlobalExplanations() {
    return this.fetch('/explanations/global');
  }

  async getAttentionVisualization(predictionId: string) {
    return this.fetch(`/explanations/attention/${predictionId}`);
  }
}

// WebSocket Client for Real-time Updates
export class MLWebSocketClient {
  private socket: Socket | null = null;
  private authToken: string | null = null;
  private tenantId: string;
  private callbacks: Map<string, Set<Function>> = new Map();

  constructor(tenantId: string, authToken?: string) {
    this.tenantId = tenantId;
    this.authToken = authToken || null;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = io(WS_URL, {
          transports: ['websocket'],
          auth: {
            tenant_id: this.tenantId,
            auth_token: this.authToken,
          },
        });

        this.socket.on('connect', () => {
          console.log('Connected to ML WebSocket server');
          resolve();
        });

        this.socket.on('connected', (data) => {
          console.log('ML WebSocket authenticated:', data);
        });

        this.socket.on('error', (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        });

        this.socket.on('disconnect', (reason) => {
          console.log('Disconnected from ML WebSocket:', reason);
        });

        // Setup event listeners
        this.setupEventListeners();
      } catch (error) {
        reject(error);
      }
    });
  }

  private setupEventListeners() {
    if (!this.socket) return;

    // Prediction updates
    this.socket.on('prediction', (data: PredictionResponse) => {
      this.emit('prediction', data);
    });

    // Drift alerts
    this.socket.on('drift_alert', (data: DriftMetrics) => {
      this.emit('drift_alert', data);
    });

    // Model updates
    this.socket.on('model_update', (data: any) => {
      this.emit('model_update', data);
    });

    // Remediation updates
    this.socket.on('remediation', (data: any) => {
      this.emit('remediation', data);
    });
  }

  subscribe(resourceIds: string[], predictionTypes: string[] = ['all']) {
    if (!this.socket) {
      throw new Error('WebSocket not connected');
    }

    this.socket.emit('subscribe', {
      resource_ids: resourceIds,
      prediction_types: predictionTypes,
    });
  }

  unsubscribe(resourceIds: string[]) {
    if (!this.socket) {
      throw new Error('WebSocket not connected');
    }

    this.socket.emit('unsubscribe', {
      resource_ids: resourceIds,
    });
  }

  requestPrediction(resourceId: string, priority: number = 1) {
    if (!this.socket) {
      throw new Error('WebSocket not connected');
    }

    this.socket.emit('request_prediction', {
      resource_id: resourceId,
      priority,
    });
  }

  on(event: string, callback: Function) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, new Set());
    }
    this.callbacks.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    const callbacks = this.callbacks.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  private emit(event: string, data: any) {
    const callbacks = this.callbacks.get(event);
    if (callbacks) {
      callbacks.forEach((callback) => callback(data));
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

// React Hooks
import { useState, useEffect, useCallback } from 'react';

export function useMLPredictions(tenantId: string, authToken?: string) {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [wsClient, setWsClient] = useState<MLWebSocketClient | null>(null);
  const [apiClient] = useState(() => new MLApiClient(authToken));

  useEffect(() => {
    // Initialize WebSocket client
    const client = new MLWebSocketClient(tenantId, authToken);
    
    client.connect()
      .then(() => {
        setWsClient(client);
        
        // Subscribe to all resources
        client.subscribe(['all']);
        
        // Setup real-time prediction listener
        client.on('prediction', (prediction: PredictionResponse) => {
          setPredictions((prev) => [prediction, ...prev.slice(0, 99)]);
        });
      })
      .catch((err) => {
        console.error('Failed to connect WebSocket:', err);
        setError(err);
      });

    // Load initial predictions
    apiClient.getAllPredictions()
      .then(setPredictions)
      .catch(setError)
      .finally(() => setLoading(false));

    return () => {
      client.disconnect();
    };
  }, [tenantId, authToken]);

  const createPrediction = useCallback(async (request: PredictionRequest) => {
    try {
      const prediction = await apiClient.createPrediction(request);
      setPredictions((prev) => [prediction, ...prev]);
      return prediction;
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, [apiClient]);

  const getRiskAssessment = useCallback(async (resourceId: string) => {
    return apiClient.getRiskAssessment(resourceId);
  }, [apiClient]);

  return {
    predictions,
    loading,
    error,
    createPrediction,
    getRiskAssessment,
    wsClient,
  };
}

export function useModelMetrics(authToken?: string) {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [apiClient] = useState(() => new MLApiClient(authToken));

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await apiClient.getModelMetrics();
        setMetrics(data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    
    // Refresh metrics every 30 seconds
    const interval = setInterval(fetchMetrics, 30000);
    
    return () => clearInterval(interval);
  }, [apiClient]);

  return { metrics, loading, error };
}

// Export singleton instances for global use
export const mlApiClient = new MLApiClient();
export default MLApiClient;