use axum::{
    extract::State,
    http::{header, Method, StatusCode},
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

mod action_orchestrator;
mod ai;
mod api;
mod approval_workflow;
mod approvals;
mod audit_chain;
mod auth;
mod azure_client;
mod azure_client_async;
mod cache;
mod change_management;
mod collectors;
mod compliance;
mod data_mode;
mod enforcement;
mod events;
mod evidence_pipeline;
mod finops;
mod observability;
mod policy_engine;
mod secret_guard;
mod secrets;
mod security_graph;
mod simulated_data;
mod slo;
mod tenant;
mod tenant_isolation;

use api::{
    create_action, create_exception, get_action, get_compliance, get_correlations, get_costs_deep,
    get_metrics, get_network_deep, get_policies, get_policies_deep, get_predictions, get_rbac_deep,
    get_recommendations, get_resources, get_resources_deep, process_conversation, remediate,
    stream_action_events, AppState,
};
use auth::{AuthUser, OptionalAuthUser};
use azure_client::AzureClient;
use azure_client_async::AsyncAzureClient;
use tenant_isolation::{tenant_isolation_middleware, TenantContext, TenantDatabase};

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    service: String,
    patents: Vec<String>,
    azure_connected: bool,
    data_mode: String,
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let azure_connected = state.async_azure_client.is_some() || state.azure_client.is_some();
    let data_mode = if azure_connected { "live" } else { "simulated" };

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
        azure_connected,
        data_mode: data_mode.to_string(),
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
            warn!(
                "⚠️ Failed to initialize fallback Azure client: {} - real Azure data unavailable",
                e
            );
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
        .route(
            "/api/v1/recommendations/proactive",
            get(get_recommendations),
        )
        // Policies endpoints
        .route("/api/v1/policies", get(get_policies))
        // Compliance and Resources endpoints
        .route("/api/v1/compliance", get(get_compliance))
        .route("/api/v1/resources", get(get_resources))
        // Deep insights (Phase 1)
        .route("/api/v1/policies/deep", get(get_policies_deep))
        .route("/api/v1/rbac/deep", get(get_rbac_deep))
        .route("/api/v1/costs/deep", get(get_costs_deep))
        .route("/api/v1/network/deep", get(get_network_deep))
        .route("/api/v1/resources/deep", get(get_resources_deep))
        // Actions (Phase 1)
        .route("/api/v1/remediate", post(remediate))
        .route("/api/v1/exception", post(create_exception))
        // Actions (Phase 2)
        .route("/api/v1/actions", post(create_action))
        .route("/api/v1/actions/:id", get(get_action))
        .route("/api/v1/actions/:id/events", get(stream_action_events))
        // Legacy endpoints for compatibility
        // Note: /api/v1/policies, /api/v1/resources and /api/v1/compliance are already registered above
        .layer(ServiceBuilder::new().layer(cors))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("PolicyCortex Core API listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
