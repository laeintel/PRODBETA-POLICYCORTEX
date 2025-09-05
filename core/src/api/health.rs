// Health Check API for PCG Platform
// Implements fail-fast pattern for real mode validation
// © 2024 PolicyCortex. All rights reserved.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::env;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};

#[derive(Debug, Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub mode: String,
    pub timestamp: DateTime<Utc>,
    pub checks: HealthChecks,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration_hints: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct HealthChecks {
    pub azure_connection: ServiceCheck,
    pub database: ServiceCheck,
    pub ml_service: ServiceCheck,
    pub cache: ServiceCheck,
    pub authentication: ServiceCheck,
}

#[derive(Debug, Serialize)]
pub struct ServiceCheck {
    pub status: String,
    pub latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl ServiceCheck {
    fn healthy(latency_ms: u64) -> Self {
        Self {
            status: "healthy".to_string(),
            latency_ms: Some(latency_ms),
            error: None,
            hint: None,
        }
    }

    fn unhealthy(error: String, hint: String) -> Self {
        Self {
            status: "unhealthy".to_string(),
            latency_ms: None,
            error: Some(error),
            hint: Some(hint),
        }
    }

    fn unknown(hint: String) -> Self {
        Self {
            status: "unknown".to_string(),
            latency_ms: None,
            error: None,
            hint: Some(hint),
        }
    }
}

// GET /api/v1/health
pub async fn get_health_status(
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    let start_time = std::time::Instant::now();
    let mut hints = Vec::new();

    info!("Performing health check - mode: {}", if state.use_real_data { "real" } else { "mock" });

    // Check Azure connection
    let azure_check = if state.use_real_data {
        if state.async_azure_client.is_some() {
            // Try to validate Azure connection
            match check_azure_connection(&state).await {
                Ok(latency) => ServiceCheck::healthy(latency),
                Err(e) => {
                    warn!("Azure connection unhealthy: {}", e);
                    ServiceCheck::unhealthy(
                        e.to_string(),
                        "Check Azure credentials: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID".to_string()
                    )
                }
            }
        } else {
            hints.push("Azure client not initialized. Set USE_REAL_DATA=true and configure Azure credentials.".to_string());
            ServiceCheck::unhealthy(
                "Azure client not configured".to_string(),
                "Set environment variables: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID".to_string()
            )
        }
    } else {
        ServiceCheck::unknown("Mock mode - Azure connection not checked".to_string())
    };

    // Check database connection
    let db_check = if let Some(ref pool) = state.db_pool {
        let db_start = std::time::Instant::now();
        match sqlx::query("SELECT 1").fetch_one(pool).await {
            Ok(_) => ServiceCheck::healthy(db_start.elapsed().as_millis() as u64),
            Err(e) => {
                error!("Database health check failed: {}", e);
                hints.push("Database connection failed. Check DATABASE_URL environment variable.".to_string());
                ServiceCheck::unhealthy(
                    format!("Database error: {}", e),
                    "Verify DATABASE_URL points to a valid PostgreSQL instance".to_string()
                )
            }
        }
    } else {
        hints.push("Database not configured. Set DATABASE_URL to enable persistence.".to_string());
        ServiceCheck::unknown("Database pool not initialized".to_string())
    };

    // Check ML service
    let ml_check = if state.use_real_data {
        if let Ok(ml_url) = env::var("PREDICTIONS_URL") {
            match check_ml_service(&ml_url).await {
                Ok(latency) => ServiceCheck::healthy(latency),
                Err(e) => {
                    warn!("ML service check failed: {}", e);
                    hints.push("ML service unreachable. Deploy models using backend/services/ai_engine/deploy_models.py".to_string());
                    ServiceCheck::unhealthy(
                        e.to_string(),
                        format!("Ensure ML service is running at {}", ml_url)
                    )
                }
            }
        } else {
            hints.push("ML service URL not configured. Set PREDICTIONS_URL to enable AI predictions.".to_string());
            ServiceCheck::unknown("PREDICTIONS_URL not set".to_string())
        }
    } else {
        ServiceCheck::unknown("Mock mode - ML service not checked".to_string())
    };

    // Check cache (Redis/DragonflyDB)
    let cache_check = if let Ok(redis_url) = env::var("REDIS_URL") {
        match check_redis(&redis_url).await {
            Ok(latency) => ServiceCheck::healthy(latency),
            Err(e) => {
                warn!("Cache service check failed: {}", e);
                hints.push("Cache service unavailable. Install Redis or DragonflyDB for better performance.".to_string());
                ServiceCheck::unhealthy(
                    e.to_string(),
                    "Redis/DragonflyDB not running. Start with: docker run -p 6379:6379 redis".to_string()
                )
            }
        }
    } else {
        ServiceCheck::unknown("REDIS_URL not configured - caching disabled".to_string())
    };

    // Check authentication
    let auth_check = if state.use_real_data {
        let required_vars = vec![
            ("AZURE_AD_TENANT_ID", env::var("AZURE_AD_TENANT_ID").is_ok()),
            ("AZURE_AD_CLIENT_ID", env::var("AZURE_AD_CLIENT_ID").is_ok()),
        ];
        
        let missing: Vec<&str> = required_vars
            .iter()
            .filter(|(_, exists)| !exists)
            .map(|(name, _)| *name)
            .collect();
            
        if missing.is_empty() {
            ServiceCheck::healthy(0)
        } else {
            hints.push(format!("Authentication not fully configured. Missing: {}", missing.join(", ")));
            ServiceCheck::unhealthy(
                "Authentication configuration incomplete".to_string(),
                format!("Set these environment variables: {}", missing.join(", "))
            )
        }
    } else {
        ServiceCheck::unknown("Mock mode - authentication bypassed".to_string())
    };

    // Determine overall status
    let all_checks = vec![&azure_check, &db_check, &ml_check, &cache_check, &auth_check];
    let overall_status = if all_checks.iter().all(|c| c.status == "healthy") {
        "healthy"
    } else if all_checks.iter().any(|c| c.status == "unhealthy") {
        "degraded"
    } else {
        "partial"
    };

    info!("Health check complete - status: {}, elapsed: {}ms", 
          overall_status, start_time.elapsed().as_millis());

    let health = HealthStatus {
        status: overall_status.to_string(),
        version: state.config.service_version.clone(),
        mode: if state.use_real_data { "real" } else { "mock" }.to_string(),
        timestamp: Utc::now(),
        checks: HealthChecks {
            azure_connection: azure_check,
            database: db_check,
            ml_service: ml_check,
            cache: cache_check,
            authentication: auth_check,
        },
        configuration_hints: if !hints.is_empty() { Some(hints) } else { None },
    };

    // Return appropriate status code based on health
    let status_code = match overall_status {
        "healthy" => StatusCode::OK,
        "degraded" => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::PARTIAL_CONTENT,
    };

    (status_code, Json(health)).into_response()
}

// GET /api/v1/health/live - Kubernetes liveness probe
pub async fn liveness_probe(
    State(_state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    // Simple check that the service is running
    (StatusCode::OK, Json(serde_json::json!({
        "status": "alive",
        "timestamp": Utc::now().to_rfc3339()
    })))
}

// GET /api/v1/health/ready - Kubernetes readiness probe
pub async fn readiness_probe(
    State(state): State<Arc<crate::api::AppState>>,
) -> impl IntoResponse {
    // Check if critical services are ready
    let ready = if state.use_real_data {
        // In real mode, require Azure connection
        state.async_azure_client.is_some()
    } else {
        // In mock mode, always ready
        true
    };

    if ready {
        (StatusCode::OK, Json(serde_json::json!({
            "status": "ready",
            "timestamp": Utc::now().to_rfc3339()
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "status": "not_ready",
            "timestamp": Utc::now().to_rfc3339()
        })))
    }
}

// Helper function to check Azure connection
async fn check_azure_connection(state: &Arc<crate::api::AppState>) -> Result<u64, String> {
    let start = std::time::Instant::now();
    
    if let Some(ref _client) = state.async_azure_client {
        // For now, if the client exists, assume it's healthy
        // In production, we'd make a real API call here
        Ok(start.elapsed().as_millis() as u64)
    } else {
        Err("Azure client not initialized".to_string())
    }
}

// Helper function to check ML service
async fn check_ml_service(url: &str) -> Result<u64, String> {
    let start = std::time::Instant::now();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
    match client.get(format!("{}/health", url)).send().await {
        Ok(response) if response.status().is_success() => {
            Ok(start.elapsed().as_millis() as u64)
        }
        Ok(response) => Err(format!("ML service returned status: {}", response.status())),
        Err(e) => Err(format!("ML service unreachable: {}", e))
    }
}

// Helper function to check Redis
async fn check_redis(url: &str) -> Result<u64, String> {
    let start = std::time::Instant::now();
    
    // Simple ping using redis crate
    match redis::Client::open(url) {
        Ok(client) => {
            let mut conn = client.get_connection()
                .map_err(|e| format!("Redis connection failed: {}", e))?;
            
            redis::cmd("PING")
                .query::<String>(&mut conn)
                .map_err(|e| format!("Redis ping failed: {}", e))?;
                
            Ok(start.elapsed().as_millis() as u64)
        }
        Err(e) => Err(format!("Invalid Redis URL: {}", e))
    }
}