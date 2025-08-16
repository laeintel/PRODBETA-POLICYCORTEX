use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::time::Duration;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetrics {
    pub request_id: String,
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub endpoint: String,
    pub method: String,
    pub status_code: u16,
    pub request_size_bytes: u64,
    pub response_size_bytes: u64,
    pub processing_time_ms: u64,
    pub timestamp: DateTime<Utc>,
    pub tier: TierType,
    pub usage_type: UsageType,
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierType {
    Free,
    Pro,
    Enterprise,
}

impl Default for TierType {
    fn default() -> Self {
        TierType::Free
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageType {
    ApiCall,
    Prediction,
    Analysis,
    Storage,
    Compute,
}

#[derive(Debug, Clone)]
pub struct QuotaLimits {
    pub api_calls_per_month: u64,
    pub predictions_per_month: u64,
    pub storage_mb_per_month: u64,
    pub compute_seconds_per_month: u64,
}

impl QuotaLimits {
    pub fn for_tier(tier: &TierType) -> Self {
        match tier {
            TierType::Free => QuotaLimits {
                api_calls_per_month: 1000,
                predictions_per_month: 100,
                storage_mb_per_month: 1024,
                compute_seconds_per_month: 3600,
            },
            TierType::Pro => QuotaLimits {
                api_calls_per_month: 50000,
                predictions_per_month: 10000,
                storage_mb_per_month: 10240,
                compute_seconds_per_month: 36000,
            },
            TierType::Enterprise => QuotaLimits {
                api_calls_per_month: 1000000,
                predictions_per_month: 100000,
                storage_mb_per_month: 102400,
                compute_seconds_per_month: 360000,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct UsageTracker {
    pub metering_service_url: String,
    pub quota_enforcement_enabled: bool,
}

impl UsageTracker {
    pub fn new(metering_service_url: String, quota_enforcement_enabled: bool) -> Self {
        Self {
            metering_service_url,
            quota_enforcement_enabled,
        }
    }

    pub async fn check_quota(&self, tenant_id: &str, usage_type: &UsageType) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if !self.quota_enforcement_enabled {
            return Ok(true);
        }

        let client = reqwest::Client::new();
        let response = client
            .get(&format!("{}/quotas/{}", self.metering_service_url, tenant_id))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        if response.status().is_success() {
            let quotas: Vec<serde_json::Value> = response.json().await?;
            
            for quota in quotas {
                if let (Some(quota_type), Some(usage_percentage)) = (
                    quota.get("usage_type").and_then(|v| v.as_str()),
                    quota.get("usage_percentage").and_then(|v| v.as_f64()),
                ) {
                    let quota_usage_type = match quota_type {
                        "api_call" => UsageType::ApiCall,
                        "prediction" => UsageType::Prediction,
                        "analysis" => UsageType::Analysis,
                        "storage" => UsageType::Storage,
                        "compute" => UsageType::Compute,
                        _ => continue,
                    };

                    if std::mem::discriminant(&quota_usage_type) == std::mem::discriminant(usage_type) {
                        return Ok(usage_percentage < 100.0);
                    }
                }
            }
        }

        // Default to allowing if we can't check quotas
        Ok(true)
    }

    pub async fn record_usage(&self, metrics: &UsageMetrics) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::new();
        
        let usage_data = serde_json::json!({
            "tenant_id": metrics.tenant_id,
            "api_endpoint": metrics.endpoint,
            "usage_type": metrics.usage_type,
            "quantity": 1.0,
            "unit": "call",
            "request_size_bytes": metrics.request_size_bytes,
            "response_size_bytes": metrics.response_size_bytes,
            "processing_time_ms": metrics.processing_time_ms,
            "timestamp": metrics.timestamp
        });

        let response = client
            .post(&format!("{}/usage/record", self.metering_service_url))
            .json(&usage_data)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if !response.status().is_success() {
            warn!("Failed to record usage: {}", response.status());
        }

        Ok(())
    }

    fn extract_tenant_id(&self, headers: &HeaderMap) -> String {
        headers
            .get("X-Tenant-ID")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("default")
            .to_string()
    }

    fn extract_user_id(&self, headers: &HeaderMap) -> Option<String> {
        headers
            .get("X-User-ID")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
    }

    fn determine_usage_type(&self, path: &str) -> UsageType {
        if path.contains("/predictions") || path.contains("/predict") {
            UsageType::Prediction
        } else if path.contains("/analysis") || path.contains("/analyze") {
            UsageType::Analysis
        } else if path.contains("/storage") || path.contains("/upload") || path.contains("/download") {
            UsageType::Storage
        } else if path.contains("/compute") || path.contains("/process") {
            UsageType::Compute
        } else {
            UsageType::ApiCall
        }
    }

    async fn get_tenant_tier(&self, tenant_id: &str) -> TierType {
        // In a real implementation, this would query the database
        // For now, default to Free tier
        TierType::Free
    }

    fn calculate_cost(&self, metrics: &UsageMetrics) -> f64 {
        let base_rates = match metrics.tier {
            TierType::Free => match metrics.usage_type {
                UsageType::ApiCall => 0.001,
                UsageType::Prediction => 0.01,
                UsageType::Analysis => 0.1,
                UsageType::Storage => 0.0001,
                UsageType::Compute => 0.01,
            },
            TierType::Pro => match metrics.usage_type {
                UsageType::ApiCall => 0.0008,
                UsageType::Prediction => 0.008,
                UsageType::Analysis => 0.08,
                UsageType::Storage => 0.00008,
                UsageType::Compute => 0.008,
            },
            TierType::Enterprise => match metrics.usage_type {
                UsageType::ApiCall => 0.0005,
                UsageType::Prediction => 0.005,
                UsageType::Analysis => 0.05,
                UsageType::Storage => 0.00005,
                UsageType::Compute => 0.005,
            },
        };

        match metrics.usage_type {
            UsageType::Compute => (metrics.processing_time_ms as f64 / 1000.0) * base_rates,
            UsageType::Storage => (metrics.request_size_bytes as f64 / (1024.0 * 1024.0)) * base_rates,
            _ => base_rates,
        }
    }
}

pub async fn usage_tracking_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let start_time = Instant::now();
    let request_id = Uuid::new_v4().to_string();
    
    // Extract request metadata
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let headers = request.headers().clone();
    
    let tenant_id = state.usage_tracker.extract_tenant_id(&headers);
    let user_id = state.usage_tracker.extract_user_id(&headers);
    let usage_type = state.usage_tracker.determine_usage_type(&path);
    
    // Check quota before processing request
    if let Err(e) = state.usage_tracker.check_quota(&tenant_id, &usage_type).await {
        error!("Failed to check quota: {}", e);
    } else if !state.usage_tracker.check_quota(&tenant_id, &usage_type).await.unwrap_or(true) {
        warn!("Quota exceeded for tenant {} on endpoint {}", tenant_id, path);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Get request size
    let request_size = request
        .headers()
        .get("content-length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    // Process request
    let response = next.run(request).await;
    
    // Calculate metrics
    let processing_time = start_time.elapsed().as_millis() as u64;
    let status_code = response.status().as_u16();
    
    // Estimate response size (in a real implementation, you'd capture the actual response body size)
    let response_size = response
        .headers()
        .get("content-length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    // Get tenant tier
    let tier = state.usage_tracker.get_tenant_tier(&tenant_id).await;

    // Create usage metrics
    let mut metrics = UsageMetrics {
        request_id,
        tenant_id: tenant_id.clone(),
        user_id,
        endpoint: path,
        method,
        status_code,
        request_size_bytes: request_size,
        response_size_bytes: response_size,
        processing_time_ms: processing_time,
        timestamp: Utc::now(),
        tier,
        usage_type,
        cost: 0.0,
    };

    // Calculate cost
    metrics.cost = state.usage_tracker.calculate_cost(&metrics);

    // Record usage asynchronously
    let usage_tracker = state.usage_tracker.clone();
    let metrics_clone = metrics.clone();
    tokio::spawn(async move {
        if let Err(e) = usage_tracker.record_usage(&metrics_clone).await {
            error!("Failed to record usage: {}", e);
        }
    });

    // Log metrics
    info!(
        request_id = %metrics.request_id,
        tenant_id = %metrics.tenant_id,
        endpoint = %metrics.endpoint,
        processing_time_ms = %metrics.processing_time_ms,
        cost = %metrics.cost,
        "Request processed"
    );

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};

    #[test]
    fn test_quota_limits_for_tier() {
        let free_limits = QuotaLimits::for_tier(&TierType::Free);
        assert_eq!(free_limits.api_calls_per_month, 1000);
        
        let pro_limits = QuotaLimits::for_tier(&TierType::Pro);
        assert_eq!(pro_limits.api_calls_per_month, 50000);
        
        let enterprise_limits = QuotaLimits::for_tier(&TierType::Enterprise);
        assert_eq!(enterprise_limits.api_calls_per_month, 1000000);
    }

    #[test]
    fn test_usage_type_determination() {
        let tracker = UsageTracker::new("http://localhost:8083".to_string(), true);
        
        assert!(matches!(tracker.determine_usage_type("/api/v1/predictions"), UsageType::Prediction));
        assert!(matches!(tracker.determine_usage_type("/api/v1/analysis"), UsageType::Analysis));
        assert!(matches!(tracker.determine_usage_type("/api/v1/storage"), UsageType::Storage));
        assert!(matches!(tracker.determine_usage_type("/api/v1/health"), UsageType::ApiCall));
    }

    #[test]
    fn test_tenant_id_extraction() {
        let tracker = UsageTracker::new("http://localhost:8083".to_string(), true);
        let mut headers = HeaderMap::new();
        headers.insert("X-Tenant-ID", HeaderValue::from_static("test-tenant"));
        
        assert_eq!(tracker.extract_tenant_id(&headers), "test-tenant");
        
        let empty_headers = HeaderMap::new();
        assert_eq!(tracker.extract_tenant_id(&empty_headers), "default");
    }

    #[test]
    fn test_cost_calculation() {
        let tracker = UsageTracker::new("http://localhost:8083".to_string(), true);
        
        let metrics = UsageMetrics {
            request_id: "test".to_string(),
            tenant_id: "test".to_string(),
            user_id: None,
            endpoint: "/test".to_string(),
            method: "GET".to_string(),
            status_code: 200,
            request_size_bytes: 1024,
            response_size_bytes: 2048,
            processing_time_ms: 100,
            timestamp: Utc::now(),
            tier: TierType::Pro,
            usage_type: UsageType::ApiCall,
            cost: 0.0,
        };
        
        let cost = tracker.calculate_cost(&metrics);
        assert_eq!(cost, 0.0008); // Pro tier API call rate
        
        let compute_metrics = UsageMetrics {
            usage_type: UsageType::Compute,
            processing_time_ms: 1000, // 1 second
            ..metrics
        };
        
        let compute_cost = tracker.calculate_cost(&compute_metrics);
        assert_eq!(compute_cost, 0.008); // 1 second * 0.008 rate
    }
}