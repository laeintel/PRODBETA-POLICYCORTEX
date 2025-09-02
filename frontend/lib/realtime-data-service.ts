/**
 * PolicyCortex Real-Time Data Service
 * Advanced data orchestration with WebSocket streaming and intelligent caching
 */

import { EventEmitter } from 'events';

export interface MetricData {
  id: string;
  timestamp: string;
  source: 'azure' | 'aws' | 'gcp' | 'on_premise' | 'hybrid';
  type: 'performance' | 'security' | 'compliance' | 'cost' | 'availability';
  name: string;
  value: number;
  unit: string;
  tags: Record<string, string>;
  metadata: Record<string, any>;
}

export interface AggregatedMetrics {
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  p50: number;
  p95: number;
  p99: number;
}

export interface PredictionResult {
  resource_id: string;
  predictions: {
    drift: {
      drift_probability: number;
      confidence: number;
      recent_mean: number;
      historical_mean: number;
      trend: 'increasing' | 'decreasing' | 'stable';
    };
    anomaly: {
      is_anomaly: boolean;
      score: number;
      z_score: number;
      expected_range: {
        min: number;
        max: number;
      };
    };
    forecast: {
      next_hour: number;
      next_day: number;
      trend: string;
    };
  };
  recommendations: string[];
  confidence: number;
}

export interface CorrelationResult {
  correlations: Array<{
    source1: string;
    source2: string;
    score: number;
    significance: 'high' | 'medium' | 'low';
  }>;
  patterns: Array<{
    pattern: string;
    description: string;
    confidence: number;
  }>;
  insights: string[];
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class IntelligentCache {
  private cache = new Map<string, CacheEntry<any>>();
  private hitRate = new Map<string, number>();
  private predictivePreload = new Set<string>();
  
  set<T>(key: string, data: T, ttl: number = 60000): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }
  
  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }
    
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }
    
    // Track hit rate for predictive caching
    this.hitRate.set(key, (this.hitRate.get(key) || 0) + 1);
    
    return entry.data;
  }
  
  invalidate(pattern?: string): void {
    if (!pattern) {
      this.cache.clear();
      return;
    }
    
    const regex = new RegExp(pattern);
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
      }
    }
  }
  
  predictiveLoad(keys: string[]): void {
    // Mark keys for predictive loading based on access patterns
    keys.forEach(key => {
      if ((this.hitRate.get(key) || 0) > 5) {
        this.predictivePreload.add(key);
      }
    });
  }
  
  getStats() {
    return {
      size: this.cache.size,
      hitRate: Array.from(this.hitRate.entries()),
      predictiveKeys: Array.from(this.predictivePreload)
    };
  }
}

export class RealTimeDataService extends EventEmitter {
  private static instance: RealTimeDataService;
  
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private subscriptions = new Set<string>();
  private cache = new IntelligentCache();
  private dataBuffer: MetricData[] = [];
  private maxBufferSize = 1000;
  private baseUrl: string;
  private wsUrl: string;
  private eventSource: EventSource | null = null;
  
  private constructor() {
    super();
    
    // Determine API URLs based on environment
    const isProduction = process.env.NODE_ENV === 'production';
    const apiHost = process.env.NEXT_PUBLIC_API_HOST || 'localhost:8001';
    
    this.baseUrl = isProduction 
      ? `https://api.policycortex.com`
      : `http://${apiHost}`;
    
    this.wsUrl = isProduction
      ? `wss://api.policycortex.com/ws`
      : `ws://${apiHost}/ws`;
    
    // Auto-connect on instantiation
    this.connect();
    
    // Setup periodic cache cleanup
    setInterval(() => this.cleanupCache(), 60000);
    
    // Setup heartbeat
    setInterval(() => this.heartbeat(), 30000);
  }
  
  static getInstance(): RealTimeDataService {
    if (!RealTimeDataService.instance) {
      RealTimeDataService.instance = new RealTimeDataService();
    }
    return RealTimeDataService.instance;
  }
  
  private connect(): void {
    try {
      this.ws = new WebSocket(this.wsUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.emit('connected');
        
        // Re-subscribe to all topics
        this.subscriptions.forEach(topic => {
          this.sendMessage({
            action: 'subscribe',
            topic
          });
        });
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected');
        this.reconnect();
      };
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      this.reconnect();
    }
  }
  
  private reconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('max_reconnect_failed');
      
      // Fall back to SSE
      this.connectSSE();
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  private connectSSE(): void {
    if (this.eventSource) {
      this.eventSource.close();
    }
    
    this.eventSource = new EventSource(`${this.baseUrl}/api/v1/stream`);
    
    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Failed to parse SSE message:', error);
      }
    };
    
    this.eventSource.onerror = () => {
      console.error('SSE connection error');
      // SSE will auto-reconnect
    };
  }
  
  private sendMessage(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
  
  private handleMessage(data: any): void {
    // Handle different message types
    if (data.id && data.timestamp) {
      // It's a metric
      this.handleMetric(data as MetricData);
    } else if (data.status) {
      // It's a status message
      this.emit('status', data);
    } else {
      // Unknown message type
      this.emit('message', data);
    }
  }
  
  private handleMetric(metric: MetricData): void {
    // Add to buffer
    this.dataBuffer.push(metric);
    if (this.dataBuffer.length > this.maxBufferSize) {
      this.dataBuffer.shift();
    }
    
    // Emit events
    this.emit('metric', metric);
    this.emit(`metric:${metric.source}`, metric);
    this.emit(`metric:${metric.type}`, metric);
    this.emit(`metric:${metric.source}:${metric.type}`, metric);
    
    // Update cache
    const cacheKey = `metric:${metric.source}:${metric.type}:${metric.name}`;
    this.cache.set(cacheKey, metric, 30000);
  }
  
  private heartbeat(): void {
    this.sendMessage({ action: 'ping' });
  }
  
  private cleanupCache(): void {
    // This is handled internally by the cache
    const stats = this.cache.getStats();
    console.log('Cache stats:', stats);
  }
  
  // Public API methods
  
  subscribe(topic: string): void {
    this.subscriptions.add(topic);
    this.sendMessage({
      action: 'subscribe',
      topic
    });
  }
  
  unsubscribe(topic: string): void {
    this.subscriptions.delete(topic);
    this.sendMessage({
      action: 'unsubscribe',
      topic
    });
  }
  
  async fetchMetrics(params?: {
    source?: string;
    type?: string;
    limit?: number;
  }): Promise<{
    metrics: MetricData[];
    aggregates: AggregatedMetrics;
    timestamp: string;
  }> {
    const cacheKey = `metrics:${JSON.stringify(params || {})}`;
    const cached = this.cache.get<any>(cacheKey);
    
    if (cached) {
      return cached;
    }
    
    try {
      const queryParams = new URLSearchParams();
      if (params?.source) queryParams.append('source', params.source);
      if (params?.type) queryParams.append('type', params.type);
      if (params?.limit) queryParams.append('limit', params.limit.toString());
      
      const response = await fetch(
        `${this.baseUrl}/api/v1/metrics?${queryParams}`,
        {
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Cache the result
      this.cache.set(cacheKey, data, 30000);
      
      return data;
      
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      
      // Return cached data if available, even if expired
      const expiredCache = this.cache.get<any>(cacheKey);
      if (expiredCache) {
        return expiredCache;
      }
      
      // Return empty data as fallback
      return {
        metrics: [],
        aggregates: {
          count: 0,
          mean: 0,
          std: 0,
          min: 0,
          max: 0,
          p50: 0,
          p95: 0,
          p99: 0
        },
        timestamp: new Date().toISOString()
      };
    }
  }
  
  async fetchPredictions(resourceId: string): Promise<PredictionResult> {
    const cacheKey = `predictions:${resourceId}`;
    const cached = this.cache.get<PredictionResult>(cacheKey);
    
    if (cached) {
      return cached;
    }
    
    try {
      const response = await fetch(
        `${this.baseUrl}/api/v1/predictions/${resourceId}`,
        {
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Cache the result
      this.cache.set(cacheKey, data, 60000);
      
      return data;
      
    } catch (error) {
      console.error('Failed to fetch predictions:', error);
      
      // Return default predictions as fallback
      return {
        resource_id: resourceId,
        predictions: {
          drift: {
            drift_probability: 0,
            confidence: 0,
            recent_mean: 0,
            historical_mean: 0,
            trend: 'stable'
          },
          anomaly: {
            is_anomaly: false,
            score: 0,
            z_score: 0,
            expected_range: {
              min: 0,
              max: 100
            }
          },
          forecast: {
            next_hour: 0,
            next_day: 0,
            trend: 'stable'
          }
        },
        recommendations: ['Unable to generate predictions at this time'],
        confidence: 0
      };
    }
  }
  
  async fetchCorrelations(params?: {
    domain?: string;
    time_range?: string;
  }): Promise<CorrelationResult> {
    const cacheKey = `correlations:${JSON.stringify(params || {})}`;
    const cached = this.cache.get<CorrelationResult>(cacheKey);
    
    if (cached) {
      return cached;
    }
    
    try {
      const queryParams = new URLSearchParams();
      if (params?.domain) queryParams.append('domain', params.domain);
      if (params?.time_range) queryParams.append('time_range', params.time_range);
      
      const response = await fetch(
        `${this.baseUrl}/api/v1/correlations?${queryParams}`,
        {
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Cache the result
      this.cache.set(cacheKey, data, 120000);
      
      return data;
      
    } catch (error) {
      console.error('Failed to fetch correlations:', error);
      
      // Return empty correlations as fallback
      return {
        correlations: [],
        patterns: [],
        insights: ['Unable to fetch correlations at this time']
      };
    }
  }
  
  getBufferedMetrics(): MetricData[] {
    return [...this.dataBuffer];
  }
  
  getLatestMetric(source?: string, type?: string): MetricData | null {
    const filtered = this.dataBuffer.filter(m => {
      if (source && m.source !== source) return false;
      if (type && m.type !== type) return false;
      return true;
    });
    
    return filtered[filtered.length - 1] || null;
  }
  
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    
    this.subscriptions.clear();
    this.cache.invalidate();
  }
  
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN || this.eventSource?.readyState === EventSource.OPEN;
  }
  
  getCacheStats() {
    return this.cache.getStats();
  }
}

// Export singleton instance
export const dataService = RealTimeDataService.getInstance();