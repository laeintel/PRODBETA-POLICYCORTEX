use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, warn, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantPlan {
    pub tier: String,
    pub requests_per_minute: u32,
    pub burst_size: u32,
    pub requests_per_day: u64,
    pub storage_gb: u32,
    pub compute_cores: u32,
}

impl TenantPlan {
    pub fn free() -> Self {
        TenantPlan {
            tier: "free".to_string(),
            requests_per_minute: 60,
            burst_size: 100,
            requests_per_day: 10000,
            storage_gb: 5,
            compute_cores: 2,
        }
    }

    pub fn pro() -> Self {
        TenantPlan {
            tier: "pro".to_string(),
            requests_per_minute: 300,
            burst_size: 500,
            requests_per_day: 100000,
            storage_gb: 20,
            compute_cores: 4,
        }
    }

    pub fn enterprise() -> Self {
        TenantPlan {
            tier: "enterprise".to_string(),
            requests_per_minute: 1000,
            burst_size: 2000,
            requests_per_day: 1000000,
            storage_gb: 100,
            compute_cores: 8,
        }
    }
}

#[derive(Clone)]
pub struct QuotaManager {
    redis_conn: Arc<RwLock<ConnectionManager>>,
    plans: Arc<HashMap<String, TenantPlan>>,
}

impl QuotaManager {
    pub async fn new(redis_url: &str) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(redis_url)?;
        let conn = ConnectionManager::new(client).await?;
        
        let mut plans = HashMap::new();
        plans.insert("free".to_string(), TenantPlan::free());
        plans.insert("pro".to_string(), TenantPlan::pro());
        plans.insert("enterprise".to_string(), TenantPlan::enterprise());
        
        Ok(QuotaManager {
            redis_conn: Arc::new(RwLock::new(conn)),
            plans: Arc::new(plans),
        })
    }

    pub async fn check_rate_limit(
        &self,
        tenant_id: &str,
        plan: &TenantPlan,
    ) -> Result<RateLimitResult, redis::RedisError> {
        let mut conn = self.redis_conn.write().await;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Token bucket keys
        let bucket_key = format!("quota:bucket:{}:tokens", tenant_id);
        let timestamp_key = format!("quota:bucket:{}:timestamp", tenant_id);
        let daily_key = format!("quota:daily:{}:{}", tenant_id, now / 86400);
        
        // Get current bucket state
        let tokens: Option<i64> = conn.get(&bucket_key).await?;
        let last_refill: Option<u64> = conn.get(&timestamp_key).await?;
        
        let (current_tokens, last_timestamp) = match (tokens, last_refill) {
            (Some(t), Some(ts)) => (t, ts),
            _ => {
                // Initialize bucket
                conn.set_ex::<_, _, ()>(&bucket_key, plan.burst_size as i64, 60).await?;
                conn.set_ex::<_, _, ()>(&timestamp_key, now, 60).await?;
                (plan.burst_size as i64, now)
            }
        };
        
        // Calculate tokens to add based on time elapsed
        let elapsed_seconds = (now - last_timestamp) as f64;
        let tokens_to_add = (elapsed_seconds * (plan.requests_per_minute as f64 / 60.0)) as i64;
        let new_tokens = (current_tokens + tokens_to_add).min(plan.burst_size as i64);
        
        // Check if request can proceed
        if new_tokens > 0 {
            // Consume a token
            conn.set_ex::<_, _, ()>(&bucket_key, new_tokens - 1, 60).await?;
            conn.set_ex::<_, _, ()>(&timestamp_key, now, 60).await?;
            
            // Increment daily counter
            let daily_count: u64 = conn.incr(&daily_key, 1).await?;
            if daily_count == 1 {
                // Set expiry for daily counter
                conn.expire::<_, ()>(&daily_key, 86400).await?;
            }
            
            // Check daily limit
            if daily_count > plan.requests_per_day {
                return Ok(RateLimitResult {
                    allowed: false,
                    remaining_tokens: 0,
                    reset_after_seconds: 86400 - (now % 86400),
                    daily_remaining: 0,
                    reason: Some("Daily quota exceeded".to_string()),
                });
            }
            
            Ok(RateLimitResult {
                allowed: true,
                remaining_tokens: (new_tokens - 1) as u32,
                reset_after_seconds: 60 - (elapsed_seconds as u64 % 60),
                daily_remaining: plan.requests_per_day - daily_count,
                reason: None,
            })
        } else {
            // Calculate when tokens will be available
            let seconds_until_token = 
                (60.0 / plan.requests_per_minute as f64).ceil() as u64;
            
            Ok(RateLimitResult {
                allowed: false,
                remaining_tokens: 0,
                reset_after_seconds: seconds_until_token,
                daily_remaining: plan.requests_per_day - 
                    conn.get::<_, u64>(&daily_key).await.unwrap_or(0),
                reason: Some("Rate limit exceeded".to_string()),
            })
        }
    }

    pub async fn get_tenant_plan(&self, tenant_id: &str) -> TenantPlan {
        let conn = self.redis_conn.read().await;
        
        // In production, fetch from database or cache
        // For now, check Redis for cached plan
        let plan_key = format!("tenant:{}:plan", tenant_id);
        
        // Clone the connection for the mutable operation
        let mut conn_clone = conn.clone();
        
        if let Ok(Some(tier)) = conn_clone.get::<_, Option<String>>(&plan_key).await {
            self.plans.get(&tier).cloned().unwrap_or_else(TenantPlan::free)
        } else {
            // Default to free plan
            TenantPlan::free()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub remaining_tokens: u32,
    pub reset_after_seconds: u64,
    pub daily_remaining: u64,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct QuotaExceededResponse {
    error: String,
    retry_after: u64,
    daily_remaining: u64,
    upgrade_url: String,
}

pub async fn quota_middleware(
    State(quota_manager): State<Arc<QuotaManager>>,
    request: Request,
    next: Next,
) -> Response {
    // Extract tenant ID from request
    let tenant_id = extract_tenant_id(&request).unwrap_or_else(|| "default".to_string());
    
    // Get tenant plan
    let plan = quota_manager.get_tenant_plan(&tenant_id).await;
    
    // Check rate limit
    match quota_manager.check_rate_limit(&tenant_id, &plan).await {
        Ok(result) => {
            if result.allowed {
                // Add rate limit headers
                let mut response = next.run(request).await;
                let headers = response.headers_mut();
                
                headers.insert(
                    "X-RateLimit-Limit",
                    plan.requests_per_minute.to_string().parse().unwrap(),
                );
                headers.insert(
                    "X-RateLimit-Remaining",
                    result.remaining_tokens.to_string().parse().unwrap(),
                );
                headers.insert(
                    "X-RateLimit-Reset",
                    result.reset_after_seconds.to_string().parse().unwrap(),
                );
                headers.insert(
                    "X-Daily-Quota-Remaining",
                    result.daily_remaining.to_string().parse().unwrap(),
                );
                
                // Emit metrics
                increment_request_counter(&tenant_id, &plan.tier, true).await;
                
                response
            } else {
                // Rate limit exceeded
                warn!(
                    "Rate limit exceeded for tenant {} (plan: {}): {}",
                    tenant_id,
                    plan.tier,
                    result.reason.as_ref().unwrap_or(&"Unknown".to_string())
                );
                
                // Emit metrics
                increment_request_counter(&tenant_id, &plan.tier, false).await;
                
                let error_response = QuotaExceededResponse {
                    error: result.reason.unwrap_or_else(|| "Rate limit exceeded".to_string()),
                    retry_after: result.reset_after_seconds,
                    daily_remaining: result.daily_remaining,
                    upgrade_url: format!("https://policycortex.com/upgrade?tenant={}", tenant_id),
                };
                
                let mut response = (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(error_response),
                ).into_response();
                
                response.headers_mut().insert(
                    "Retry-After",
                    result.reset_after_seconds.to_string().parse().unwrap(),
                );
                response.headers_mut().insert(
                    "X-RateLimit-Reset",
                    result.reset_after_seconds.to_string().parse().unwrap(),
                );
                
                response
            }
        }
        Err(e) => {
            // Redis error - fail open for now
            warn!("Failed to check rate limit: {}. Allowing request.", e);
            next.run(request).await
        }
    }
}

fn extract_tenant_id(request: &Request) -> Option<String> {
    // Try to extract from JWT claims
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                // In production, decode JWT and extract tenant_id claim
                // For now, use a placeholder
                return Some("tenant_123".to_string());
            }
        }
    }
    
    // Try to extract from X-Tenant-ID header
    if let Some(tenant_header) = request.headers().get("X-Tenant-ID") {
        if let Ok(tenant_id) = tenant_header.to_str() {
            return Some(tenant_id.to_string());
        }
    }
    
    // Try to extract from path
    let path = request.uri().path();
    if path.starts_with("/api/v1/tenants/") {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() > 4 {
            return Some(parts[4].to_string());
        }
    }
    
    None
}

async fn increment_request_counter(tenant_id: &str, plan_tier: &str, allowed: bool) {
    // In production, this would update Prometheus metrics
    // For now, just log
    debug!(
        "Request counter: tenant={}, plan={}, allowed={}",
        tenant_id, plan_tier, allowed
    );
}

// Usage metering for billing
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageEvent {
    pub tenant_id: String,
    pub timestamp: u64,
    pub event_type: String,
    pub resource_type: String,
    pub quantity: f64,
    pub metadata: HashMap<String, String>,
}

pub struct UsageTracker {
    event_grid_endpoint: String,
    redis_conn: Arc<RwLock<ConnectionManager>>,
}

impl UsageTracker {
    pub async fn new(redis_url: &str, event_grid_endpoint: String) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(redis_url)?;
        let conn = ConnectionManager::new(client).await?;
        
        Ok(UsageTracker {
            event_grid_endpoint,
            redis_conn: Arc::new(RwLock::new(conn)),
        })
    }

    pub async fn track_usage(&self, event: UsageEvent) -> Result<(), Box<dyn std::error::Error>> {
        // Buffer events in Redis
        let mut conn = self.redis_conn.write().await;
        let queue_key = "usage:events:queue";
        
        let event_json = serde_json::to_string(&event)?;
        conn.rpush::<_, _, ()>(queue_key, event_json).await?;
        
        // Batch publish to Event Grid every 100 events or 60 seconds
        let queue_length: usize = conn.llen(queue_key).await?;
        if queue_length >= 100 {
            self.flush_events().await?;
        }
        
        Ok(())
    }

    async fn flush_events(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut conn = self.redis_conn.write().await;
        let queue_key = "usage:events:queue";
        
        // Get all events
        let events: Vec<String> = conn.lrange(queue_key, 0, -1).await?;
        if events.is_empty() {
            return Ok(());
        }
        
        // Parse events
        let usage_events: Vec<UsageEvent> = events
            .iter()
            .filter_map(|e| serde_json::from_str(e).ok())
            .collect();
        
        // Send to Event Grid
        let client = reqwest::Client::new();
        let response = client
            .post(&self.event_grid_endpoint)
            .json(&usage_events)
            .send()
            .await?;
        
        if response.status().is_success() {
            // Clear queue on success
            conn.del::<_, ()>(queue_key).await?;
            info!("Flushed {} usage events to Event Grid", usage_events.len());
        } else {
            warn!("Failed to send usage events: {}", response.status());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_plans() {
        let free = TenantPlan::free();
        assert_eq!(free.requests_per_minute, 60);
        assert_eq!(free.burst_size, 100);
        
        let pro = TenantPlan::pro();
        assert_eq!(pro.requests_per_minute, 300);
        assert_eq!(pro.burst_size, 500);
        
        let enterprise = TenantPlan::enterprise();
        assert_eq!(enterprise.requests_per_minute, 1000);
        assert_eq!(enterprise.burst_size, 2000);
    }
}