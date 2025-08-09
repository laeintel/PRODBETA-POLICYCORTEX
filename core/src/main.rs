use axum::{
    extract::State,
    http::{Method, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod auth;
mod azure_client;
mod azure_client_async;
mod cache;

use api::{
    AppState,
    get_metrics,
    get_predictions,
    get_recommendations,
    process_conversation,
    get_correlations,
    get_policies_deep,
    remediate,
    create_exception,
};
use auth::{AuthUser, OptionalAuthUser};
use azure_client::AzureClient;
use azure_client_async::AsyncAzureClient;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    service: String,
    patents: Vec<String>,
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "2.0.0".to_string(),
        service: "policycortex-core".to_string(),
        patents: vec![
            "Unified AI-Driven Cloud Governance Platform".to_string(),
            "Predictive Policy Compliance Engine".to_string(),
            "Conversational Governance Intelligence System".to_string(),
            "Cross-Domain Governance Correlation Engine".to_string(),
        ],
    })
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "policycortex_core=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting PolicyCortex v2 Core Service");
    info!("Patents: Unified AI Platform | Predictive Compliance | Conversational Intelligence | Cross-Domain Correlation");

    // Initialize high-performance async Azure client first
    let async_azure_client = match AsyncAzureClient::new().await {
        Ok(client) => {
            info!("🚀 High-performance async Azure client initialized - ultra-fast data access enabled");
            Some(client)
        }
        Err(e) => {
            warn!("⚠️ Failed to initialize async Azure client: {}", e);
            None
        }
    };

    // Initialize fallback Azure client
    let azure_client = match AzureClient::new().await {
        Ok(client) => {
            info!("✅ Fallback Azure client initialized");
            Some(client)
        }
        Err(e) => {
            warn!("⚠️ Failed to initialize fallback Azure client: {} - using mock data only", e);
            None
        }
    };

    // Initialize application state with both clients
    let mut app_state = AppState::new();
    app_state.async_azure_client = async_azure_client;
    app_state.azure_client = azure_client;
    let app_state = Arc::new(app_state);

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    // Build the application router
    let app = Router::new()
        // Health check
        .route("/health", get(health_check))
        
        // Patent 1: Unified AI Platform endpoints
        .route("/api/v1/metrics", get(get_metrics))
        .route("/api/v1/governance/unified", get(get_metrics))
        
        // Patent 2: Predictive Compliance endpoints
        .route("/api/v1/predictions", get(get_predictions))
        .route("/api/v1/compliance/predict", get(get_predictions))
        
        // Patent 3: Conversational Intelligence endpoints
        .route("/api/v1/conversation", post(process_conversation))
        .route("/api/v1/nlp/query", post(process_conversation))
        
        // Patent 4: Cross-Domain Correlation endpoints
        .route("/api/v1/correlations", get(get_correlations))
        .route("/api/v1/analysis/cross-domain", get(get_correlations))
        
        // Proactive Recommendations
        .route("/api/v1/recommendations", get(get_recommendations))
        .route("/api/v1/recommendations/proactive", get(get_recommendations))

        // Deep insights (Phase 1)
        .route("/api/v1/policies/deep", get(get_policies_deep))

        // Actions (Phase 1)
        .route("/api/v1/remediate", post(remediate))
        .route("/api/v1/exception", post(create_exception))
        
        // Legacy endpoints for compatibility
        .route("/api/v1/policies", get(get_policies))
        .route("/api/v1/resources", get(get_resources))
        .route("/api/v1/compliance", get(get_compliance))
        
        .layer(ServiceBuilder::new().layer(cors))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("PolicyCortex Core API listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// Legacy endpoint handlers
async fn get_policies(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let metrics = state.metrics.read().await;
    Json(serde_json::json!({
        "policies": [
            {
                "id": "pol-001",
                "name": "Require HTTPS",
                "description": "All web apps must use HTTPS",
                "category": "Security",
                "severity": "High",
                "status": "Active",
                "compliance_rate": metrics.policies.compliance_rate,
                "automated": true
            },
            {
                "id": "pol-002",
                "name": "Tag Compliance",
                "description": "Resources must have required tags",
                "category": "Governance",
                "severity": "Medium",
                "status": "Active",
                "compliance_rate": 95.2,
                "automated": true
            },
            {
                "id": "pol-003",
                "name": "Allowed Locations",
                "description": "Resources must be in approved regions",
                "category": "Compliance",
                "severity": "High",
                "status": "Active",
                "compliance_rate": 100.0,
                "automated": true
            }
        ],
        "total": metrics.policies.total,
        "active": metrics.policies.active,
        "violations": metrics.policies.violations
    }))
}

async fn get_resources(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let metrics = state.metrics.read().await;
    Json(serde_json::json!({
        "resources": [
            {
                "id": "res-001",
                "name": "policycortex-prod-vm",
                "type": "Microsoft.Compute/virtualMachines",
                "location": "East US",
                "status": "Optimized",
                "tags": {"Environment": "Production", "Owner": "AeoliTech"}
            },
            {
                "id": "res-002",
                "name": "policycortex-storage",
                "type": "Microsoft.Storage/storageAccounts",
                "location": "East US",
                "status": "Compliant",
                "tags": {"Environment": "Production", "Encrypted": "true"}
            }
        ],
        "total": metrics.resources.total,
        "optimized": metrics.resources.optimized,
        "idle": metrics.resources.idle
    }))
}

async fn get_compliance(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let metrics = state.metrics.read().await;
    Json(serde_json::json!({
        "compliance": {
            "overall_rate": metrics.policies.compliance_rate,
            "policies_evaluated": metrics.policies.total,
            "resources_scanned": metrics.resources.total,
            "violations_found": metrics.policies.violations,
            "auto_remediated": 8,
            "prediction_accuracy": metrics.policies.prediction_accuracy
        },
        "by_category": {
            "Security": 99.2,
            "Governance": 98.5,
            "Compliance": 100.0,
            "Cost": 94.3
        }
    }))
}