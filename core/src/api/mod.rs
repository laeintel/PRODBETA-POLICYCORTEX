// PCG Platform API modules
// Focus on three pillars: PREVENT, PROVE, PAYBACK
// © 2024 PolicyCortex. All rights reserved.

// Core modules for PCG pillars
pub mod resources;
pub mod predictions;
pub mod correlations;
pub mod ml;
pub mod evidence;
pub mod health;

// Re-export PREVENT pillar functions (Predictive Compliance)
pub use predictions::{
    get_violation_predictions, get_resource_predictions, get_risk_score, remediate_prediction
};

// Re-export PROVE pillar functions (Evidence Chain)
pub use evidence::{
    collect_evidence, verify_evidence, generate_report, get_chain_status, 
    get_evidence_by_id, get_block, get_merkle_proof
};

// Re-export PAYBACK pillar functions (ROI & Cost Optimization)
pub use correlations::{
    get_correlations, analyze_correlations, what_if_analysis, 
    get_real_time_insights, get_correlation_graph
};

// Re-export resource management functions
pub use resources::{
    get_all_resources, get_resources_by_category, get_resource_by_id,
    execute_resource_action, get_resource_insights, get_resource_health_summary
};

use crate::auth::{AuthUser, OptionalAuthUser, TenantContext, TokenValidator};
use crate::error::ApiError;
use crate::validation::Validator;
use crate::secrets::SecretsManager;
use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response, Json
    },
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::env;
use tokio::sync::RwLock;
use tracing::{error, info, warn, debug};
use uuid::Uuid;

// Application state for PCG platform
pub struct AppState {
    pub config: crate::config::AppConfig,
    pub async_azure_client: Option<crate::azure_client_async::AsyncAzureClient>,
    pub prometheus: Option<metrics_exporter_prometheus::PrometheusHandle>,
    pub secrets: Option<crate::secrets::SecretsManager>,
    pub db_pool: Option<sqlx::PgPool>,
    pub tenant_cache: RwLock<HashMap<String, TenantContext>>,
    pub token_validator: TokenValidator,
    pub validator: Validator,
    pub use_real_data: bool,
}

impl AppState {
    pub fn new() -> Self {
        let use_real_data = env::var("USE_REAL_DATA")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
            
        if use_real_data {
            info!("Real mode enabled - will connect to live Azure services");
        } else {
            info!("Mock mode enabled - using simulated data");
        }
        
        Self {
            config: crate::config::AppConfig::load(),
            async_azure_client: None,
            prometheus: None,
            secrets: None,
            db_pool: None,
            tenant_cache: RwLock::new(HashMap::new()),
            token_validator: TokenValidator::new(),
            validator: Validator::new(),
            use_real_data,
        }
    }
    
    // Helper method to check real mode and return fail-fast error
    pub fn require_real_mode(&self, service_name: &str) -> Result<(), ApiError> {
        if !self.use_real_data {
            return Err(ApiError::ServiceUnavailable {
                service: service_name.to_string(),
                hint: format!(
                    "Real mode is disabled. Set USE_REAL_DATA=true and configure: \n\
                    - AZURE_SUBSCRIPTION_ID\n\
                    - AZURE_TENANT_ID\n\
                    - AZURE_CLIENT_ID\n\
                    - AZURE_CLIENT_SECRET\n\
                    See docs/REVAMP/REAL_MODE_SETUP.md for configuration details."
                ),
            });
        }
        
        if self.async_azure_client.is_none() {
            return Err(ApiError::ServiceUnavailable {
                service: service_name.to_string(),
                hint: "Azure client not initialized. Check Azure credentials and connectivity.".to_string(),
            });
        }
        
        Ok(())
    }
}

// Core configuration endpoint
pub async fn get_config(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "platform": "Predictive Cloud Governance (PCG)",
        "version": state.config.service_version,
        "pillars": {
            "prevent": "Predictive Policy Compliance",
            "prove": "Immutable Evidence Chain", 
            "payback": "ROI & Cost Optimization"
        },
        "features": {
            "ml_enabled": true,
            "evidence_chain": true,
            "cost_optimization": true,
            "real_time_predictions": true
        }
    }))
}

// Prometheus metrics export
pub async fn export_prometheus(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(ref prometheus) = state.prometheus {
        return (StatusCode::OK, prometheus.render()).into_response();
    }
    (StatusCode::SERVICE_UNAVAILABLE, "Metrics not available").into_response()
}

// Legacy imports for compatibility - will be phased out
pub async fn get_predictions(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Redirect to new predictive compliance endpoint
    predictions::get_violation_predictions(State(state)).await
}

pub async fn get_evidence_pack(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Redirect to evidence chain status
    evidence::get_chain_status(State(state)).await
}

pub async fn get_correlations_legacy(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Legacy correlation endpoint - redirect to ROI correlations
    correlations::get_correlations(State(state)).await
}

pub async fn get_resources(State(state): State<Arc<AppState>>) -> Json<Value> {
    resources::get_all_resources(State(state)).await
}

pub async fn get_compliance(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Basic compliance status for evidence requirements
    Json(json!({
        "status": "compliant",
        "score": 94.5,
        "violations": 3,
        "policies_evaluated": 127,
        "last_scan": "2025-01-15T08:00:00Z"
    }))
}

pub async fn get_policies(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Basic policy list for compliance checks
    Json(json!({
        "policies": [
            {
                "id": "pol-001",
                "name": "Data Encryption at Rest",
                "type": "security",
                "status": "active",
                "compliance_rate": 98.5
            },
            {
                "id": "pol-002", 
                "name": "Cost Optimization Thresholds",
                "type": "financial",
                "status": "active",
                "compliance_rate": 92.0
            },
            {
                "id": "pol-003",
                "name": "Resource Tagging Standards",
                "type": "governance", 
                "status": "active",
                "compliance_rate": 88.5
            }
        ],
        "total_count": 127,
        "active_count": 125,
        "enforcement_mode": "audit"
    }))
}